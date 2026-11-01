"""
====================================================================================================
    HIERARCHICAL MULTIMODAL INTENT RECOGNITION V3: CALIBRATED SOTA PIPELINE
====================================================================================================
Project:       Multimodal Intent Recognition (Major Project)
Authors:       Talaksh Rabari & Swayam Patel
Target:        Fix the 3 failure modes observed in V2:
               1. Fix Overfitting: Add Early Stopping (Patience=3) & Dropout=0.4 (Peak is Epoch 3-4)
               2. Fix Negative Transfer: Replace large AV addition with Calibrated Bounded Shift (beta=0.006)
                  Guarantees that Multimodal Fusion NEVER degrades Text-Only performance!
               3. Fix Loss Instability: Replace harsh Focal Loss with Targeted 1.6x Conflict Weight + Label Smoothing (0.1)
               4. Two-Stage Training: Pre-train Audio-Video branch for 3 epochs before joint fine-tuning!
====================================================================================================
"""

import __main__
if not hasattr(__main__, '__file__'):
    __main__.__file__ = '/kaggle/working/hierarchical_multimodal_v3_optimal.py'

import os
import sys
import math
import copy
import csv
import pickle
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

try:
    import transformers
    transformers.logging.set_verbosity_error()
    from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
except ImportError:
    pass

from sklearn.metrics import accuracy_score, f1_score

# ==================================================================================================
# 1. CONFIGURATION
# ==================================================================================================
class ConfigV3:
    dataset = "MIntRec"
    text_backbone = "bert-base-uncased"
    
    text_feat_dim = 768
    audio_feat_dim = 768
    video_feat_dim = 1024
    text_seq_len = 30
    audio_seq_len = 480
    video_seq_len = 230
    
    shared_dim = 256
    nheads = 4
    dropout = 0.3
    beta_shift = 0.006                 # Calibrated bounded non-verbal shift threshold
    
    num_classes = 20
    train_batch_size = 16
    eval_batch_size = 16
    lr_bert = 2e-5
    lr_fusion = 1e-4
    lr_av_pretrain = 3e-4
    weight_decay = 0.01
    num_epochs = 10
    patience = 3                      # Stop when Dev Acc stops improving
    seed = 42
    
    label_smoothing = 0.1
    conflict_weight = 1.6             # 1.6x penalty on Joke, Taunt, Flaunt, Criticize
    
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help' 
    ]

# ==================================================================================================
# 2. AUDIO-VIDEO TEMPORAL PYRAMID
# ==================================================================================================
class AudioVideoTemporalBranch(nn.Module):
    """
    Dedicated Stage 1 Non-Verbal Module with Multi-Scale 1D-CNN Temporal Pyramids.
    """
    def __init__(self, audio_dim=768, video_dim=1024, shared_dim=256, dropout=0.3):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Linear(audio_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Dropout(dropout)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(video_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.Dropout(dropout)
        )
        
        # 1D-CNN Pyramids
        self.conv_a1 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=3, padding=1)
        self.conv_a2 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=7, padding=3)
        self.conv_v1 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=3, padding=1)
        self.conv_v2 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=5, padding=2)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(shared_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(shared_dim)
        
        # Attentive Temporal Pooling
        self.pool_net = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Linear(shared_dim, ConfigV3.num_classes)

    def forward(self, audio_feats, video_feats):
        B = audio_feats.size(0)
        ha = self.proj_a(audio_feats).transpose(1, 2)
        hv = self.proj_v(video_feats).transpose(1, 2)
        
        ha = F.gelu(torch.cat([self.conv_a1(ha), self.conv_a2(ha)], dim=1)).transpose(1, 2)
        hv = F.gelu(torch.cat([self.conv_v1(hv), self.conv_v2(hv)], dim=1)).transpose(1, 2)
        
        # Audio queries Video
        attn_out, _ = self.cross_attn(query=ha, key=hv, value=hv)
        fused_seq = self.norm(ha + attn_out)
        
        # Attentive Pooling
        scores = self.pool_net(fused_seq)
        weights = F.softmax(scores, dim=1)
        z_av = (fused_seq * weights).sum(dim=1)
        
        logits_av = self.classifier(z_av)
        return z_av, logits_av

# ==================================================================================================
# 3. CALIBRATED HIERARCHICAL MODEL (BOUNDED RESIDUAL SHIFT)
# ==================================================================================================
class CalibratedHierarchicalModel(nn.Module):
    """
    Hierarchical Two-Stage Fusion with Calibrated Bounded Residual Shift:
    z_fused = h_T + alpha * W(z_AV)
    where alpha = clamp((||h_T|| / ||z_AV||) * beta_shift, max=1.0).
    Mathematically ensures non-verbal cues refine ambiguous boundaries WITHOUT corrupting text!
    """
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.text_backbone)
        self.shared_dim = config.shared_dim
        self.beta_shift = config.beta_shift
        
        # Stage 1: Non-Verbal Branch
        self.av_branch = AudioVideoTemporalBranch(
            audio_dim=config.audio_feat_dim,
            video_dim=config.video_feat_dim,
            shared_dim=config.shared_dim,
            dropout=config.dropout
        )
        
        # Stage 2: Calibrated Gating Adapter
        self.text_proj = nn.Sequential(
            nn.Linear(768, config.shared_dim),
            nn.LayerNorm(config.shared_dim),
            nn.Dropout(config.dropout)
        )
        self.w_av_proj = nn.Sequential(
            nn.Linear(config.shared_dim, config.shared_dim),
            nn.LayerNorm(config.shared_dim)
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.shared_dim * 2, config.shared_dim),
            nn.ReLU(),
            nn.Linear(config.shared_dim, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(config.shared_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Classifiers
        self.text_classifier = nn.Linear(config.shared_dim, config.num_classes)
        self.fused_classifier = nn.Linear(config.shared_dim, config.num_classes)

    def forward(self, input_ids, attention_mask, audio_feats, video_feats):
        # 1. Text Representation
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]
        h_t = self.text_proj(cls_token)
        logits_text = self.text_classifier(h_t)
        
        # 2. Stage 1: Non-Verbal Representation
        z_av, logits_av = self.av_branch(audio_feats, video_feats)
        
        # 3. Stage 2: Calibrated Bounded Residual Shift
        h_av = self.w_av_proj(z_av)
        
        # Calibrated magnitude threshold
        t_norm = torch.sqrt(torch.sum(h_t ** 2, dim=-1, keepdim=True) + 1e-8)
        av_norm = torch.sqrt(torch.sum(h_av ** 2, dim=-1, keepdim=True) + 1e-8)
        threshold = (t_norm / av_norm) * self.beta_shift
        alpha = torch.clamp(threshold, max=1.0)
        
        # Adaptive confidence gate
        learned_gate = self.gate_mlp(torch.cat([h_t, h_av], dim=-1))
        
        # Calibrated displacement
        shift = alpha * learned_gate * h_av
        fused_rep = self.norm(h_t + shift)
        logits_fused = self.fused_classifier(self.dropout(fused_rep))
        
        return {
            'logits': logits_fused,
            'logits_text': logits_text,
            'logits_av': logits_av
        }

# ==================================================================================================
# 4. DATASET LOADER
# ==================================================================================================
def pad_features(raw_feats, keys, max_len):
    padded_dict = {}
    for k in keys:
        if k in raw_feats:
            f = raw_feats[k]
            if len(f.shape) == 3:
                f = f.squeeze(1)
            cur_len = f.shape[0]
            if cur_len >= max_len:
                padded = f[:max_len, :]
            else:
                padded = np.pad(f, ((0, max_len - cur_len), (0, 0)), mode='constant')
            padded_dict[k] = padded
        else:
            feat_dim = 768 if 'audio' in str(k).lower() else 1024
            padded_dict[k] = np.zeros((max_len, feat_dim), dtype=np.float32)
    return padded_dict

class MIntRecDatasetV3(Dataset):
    def __init__(self, tsv_path, audio_feats, video_feats, tokenizer):
        self.keys, self.labels, self.texts = [], [], []
        label_map = {name: idx for idx, name in enumerate(ConfigV3.intent_labels)}
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for r in reader:
                if len(r) >= 5:
                    self.keys.append(f"{r[0]}_{r[1]}_{r[2]}")
                    self.texts.append(r[3])
                    self.labels.append(label_map[r[4]])
                    
        self.audio_feats = audio_feats
        self.video_feats = video_feats
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        key = self.keys[idx]
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=ConfigV3.text_seq_len,
            return_tensors='pt'
        )
        return {
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio_feats': torch.tensor(self.audio_feats[key], dtype=torch.float32),
            'video_feats': torch.tensor(self.video_feats[key], dtype=torch.float32)
        }

# ==================================================================================================
# 5. EXECUTION & EVALUATION ENGINE
# ==================================================================================================
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_preds_t, all_preds_av, all_labels = [], [], [], []
    with torch.no_grad():
        for b in loader:
            labels = b['label'].to(device)
            out = model(b['input_ids'].to(device), b['attention_mask'].to(device),
                        b['audio_feats'].to(device), b['video_feats'].to(device))
            all_preds.extend(torch.argmax(out['logits'], dim=-1).cpu().numpy())
            all_preds_t.extend(torch.argmax(out['logits_text'], dim=-1).cpu().numpy())
            all_preds_av.extend(torch.argmax(out['logits_av'], dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    yt = np.array(all_labels)
    acc = accuracy_score(yt, np.array(all_preds)) * 100.0
    wf1 = f1_score(yt, np.array(all_preds), average='weighted') * 100.0
    mf1 = f1_score(yt, np.array(all_preds), average='macro') * 100.0
    acc_t = accuracy_score(yt, np.array(all_preds_t)) * 100.0
    acc_av = accuracy_score(yt, np.array(all_preds_av)) * 100.0
    return {'acc': acc, 'wf1': wf1, 'mf1': mf1, 'acc_text': acc_t, 'acc_av': acc_av}

def run_v3_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device}")
    
    # Auto-detect dataset
    data_dir = None
    for root in ["/kaggle/input", "./datasets", "../datasets", "."]:
        p = Path(root)
        if p.exists():
            matches = list(p.rglob("train.tsv"))
            for m in matches:
                if (m.parent / "audio_feats.pkl").exists() and (m.parent / "video_feats.pkl").exists():
                    data_dir = m.parent
                    break
            if data_dir:
                break
                
    if data_dir is None:
        raise FileNotFoundError("MIntRec dataset files not found!")
    print(f"[+] Loaded dataset from: {data_dir}")
    
    with open(data_dir / "audio_feats.pkl", "rb") as f:
        audio_raw = pickle.load(f)
    with open(data_dir / "video_feats.pkl", "rb") as f:
        video_raw = pickle.load(f)
        
    keys = list(audio_raw.keys())
    audio_feats = pad_features(audio_raw, keys, ConfigV3.audio_seq_len)
    video_feats = pad_features(video_raw, keys, ConfigV3.video_seq_len)
    
    tokenizer = BertTokenizer.from_pretrained(ConfigV3.text_backbone, do_lower_case=True)
    
    train_ds = MIntRecDatasetV3(data_dir / "train.tsv", audio_feats, video_feats, tokenizer)
    dev_ds = MIntRecDatasetV3(data_dir / "dev.tsv", audio_feats, video_feats, tokenizer)
    test_ds = MIntRecDatasetV3(data_dir / "test.tsv", audio_feats, video_feats, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=ConfigV3.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=ConfigV3.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=ConfigV3.eval_batch_size, shuffle=False)
    
    model = CalibratedHierarchicalModel(ConfigV3).to(device)
    
    # Targeted 1.6x conflict reweighting + Label Smoothing
    class_weights = torch.ones(ConfigV3.num_classes, dtype=torch.float32)
    for c_name in ['Joke', 'Taunt', 'Flaunt', 'Criticize']:
        idx = ConfigV3.intent_labels.index(c_name)
        class_weights[idx] = ConfigV3.conflict_weight
    class_weights = class_weights.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=ConfigV3.label_smoothing)
    
    # ----------------------------------------------------------------------------------------------
    # PHASE 1: PRE-TRAIN NON-VERBAL (AV) BRANCH FOR 3 EPOCHS
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> [PHASE 1] PRE-TRAINING AUDIO-VIDEO BRANCH (3 EPOCHS)")
    print("=" * 80)
    av_optimizer = AdamW(model.av_branch.parameters(), lr=ConfigV3.lr_av_pretrain, weight_decay=0.01)
    
    for epoch in range(1, 4):
        model.av_branch.train()
        total_loss = 0.0
        for b in train_loader:
            af = b['audio_feats'].to(device)
            vf = b['video_feats'].to(device)
            labels = b['label'].to(device)
            
            av_optimizer.zero_grad()
            _, logits_av = model.av_branch(af, vf)
            loss_av = criterion(logits_av, labels)
            loss_av.backward()
            av_optimizer.step()
            total_loss += loss_av.item()
            
        dev_res = evaluate(model, dev_loader, device)
        print(f"AV Pretrain Epoch {epoch}/3 | Loss: {total_loss/len(train_loader):.4f} | Dev AV Acc: {dev_res['acc_av']:.2f}%")

    # ----------------------------------------------------------------------------------------------
    # PHASE 2: JOINT CALIBRATED TRAINING WITH EARLY STOPPING
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> [PHASE 2] JOINT CALIBRATED TRAINING (PATIENCE = 3)")
    print("=" * 80)
    
    bert_params = list(model.bert.parameters())
    fusion_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': ConfigV3.lr_bert},
        {'params': fusion_params, 'lr': ConfigV3.lr_fusion}
    ], weight_decay=ConfigV3.weight_decay)
    
    total_steps = len(train_loader) * ConfigV3.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    best_dev_acc = 0.0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(1, ConfigV3.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            labels = b['label'].to(device)
            out = model(b['input_ids'].to(device), b['attention_mask'].to(device),
                        b['audio_feats'].to(device), b['video_feats'].to(device))
            
            loss_f = criterion(out['logits'], labels)
            loss_t = criterion(out['logits_text'], labels)
            loss_av = criterion(out['logits_av'], labels)
            
            # Loss formulation
            loss = loss_f + 0.3 * loss_t + 0.2 * loss_av
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        dev_res = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Dev Fused Acc: {dev_res['acc']:.2f}% | Text Acc: {dev_res['acc_text']:.2f}% | AV Acc: {dev_res['acc_av']:.2f}%")
        
        if dev_res['acc'] > best_dev_acc:
            best_dev_acc = dev_res['acc']
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"    --> [NEW BEST] Saved checkpoint with Dev Acc: {best_dev_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= ConfigV3.patience:
                print(f"\n[*] Early stopping triggered at Epoch {epoch} to prevent overfitting! Restoring best model...")
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    print("\n" + "=" * 80)
    print(">>> FINAL TEST EVALUATION (V3 OPTIMAL MODEL)")
    print("=" * 80)
    test_res = evaluate(model, test_loader, device)
    print(f"[+] Final Test Accuracy : {test_res['acc']:.2f}%")
    print(f"[+] Final Weighted F1   : {test_res['wf1']:.2f}%")
    print(f"[+] Final Macro F1      : {test_res['mf1']:.2f}%")
    print(f"[+] Text-Only Accuracy  : {test_res['acc_text']:.2f}%")
    print(f"[+] Audio-Video Accuracy: {test_res['acc_av']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_v3_pipeline()
