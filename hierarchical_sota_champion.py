"""
====================================================================================================
    HIERARCHICAL SOTA CHAMPION: BEATING THE 73.62% BENCHMARK (74.5% - 76.0% TARGET)
====================================================================================================
Project:       Multimodal Intent Recognition (Major Project)
Authors:       Talaksh Rabari & Swayam Patel (Dhirubhai Ambani University)
Advisor:       Dr. Shruti Bhilare
Goal:          Outperform the official AAAI 2024 Benchmark (73.62% Acc / 73.31% WF1)
Architecture:  - Professor's Hierarchical Two-Stage Fusion (A+V -> then + Text)
               - UAGM (Uncertainty-Aware Sequence-Variance Noise Filtering from DUMIR)
               - Multi-Scale 1D-CNN Temporal Pyramids
               - Calibrated Bounded Shift Gate (beta = 0.007)
               - Targeted 1.6x Conflict Reweighting + Label Smoothing
               - Multi-Seed Evaluation (Seeds 0, 1, 2) + Softmax Ensemble Arbitration
====================================================================================================
"""

import __main__
if not hasattr(__main__, '__file__'):
    __main__.__file__ = '/kaggle/working/hierarchical_sota_champion.py'

import os
import sys
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
class SOTAConfig:
    dataset = "MIntRec"
    text_backbone = "bert-base-uncased"
    
    text_feat_dim = 768
    audio_feat_dim = 768
    video_feat_dim = 1024
    text_seq_len = 30
    audio_seq_len = 480
    video_seq_len = 230
    
    shared_dim = 256
    dropout = 0.3
    beta_shift = 0.007                 # Calibrated bounded non-verbal shift
    alpha_uncertainty = 4.0            # DUMIR sequence variance penalty
    
    num_classes = 20
    train_batch_size = 16
    eval_batch_size = 16
    lr_bert = 2e-5
    lr_fusion = 1e-4
    lr_av_pretrain = 3e-4
    weight_decay = 0.01
    
    epochs_av = 3
    epochs_joint = 10
    patience = 3                       # Early stopping patience
    
    seeds = [0, 1, 2]                  # 3-Seed Benchmark & Softmax Ensemble
    label_smoothing = 0.1
    conflict_weight = 1.6
    
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help' 
    ]

# ==================================================================================================
# 2. DUMIR UNCERTAINTY-AWARE NOISE FILTER (UAGM)
# ==================================================================================================
class UncertaintyNoiseFilter(nn.Module):
    """
    Estimates sequence-level variance across time to filter out blurry video frames & silent audio.
    """
    def __init__(self, in_dim, hidden_dim=128, alpha=4.0):
        super().__init__()
        self.alpha = alpha
        self.saliency_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, feats):
        # feats: (B, L, D)
        mean_centroid = torch.mean(feats, dim=1, keepdim=True)
        var = torch.mean((feats - mean_centroid) ** 2, dim=-1, keepdim=True)  # (B, L, 1)
        saliency = self.saliency_mlp(feats)                                    # (B, L, 1)
        reliability = saliency * torch.exp(-self.alpha * var)
        return feats * reliability

# ==================================================================================================
# 3. STAGE 1: NON-VERBAL (AUDIO-VIDEO) TEMPORAL PYRAMID
# ==================================================================================================
class AudioVideoTemporalPyramid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.u_audio = UncertaintyNoiseFilter(config.audio_feat_dim, alpha=config.alpha_uncertainty)
        self.u_video = UncertaintyNoiseFilter(config.video_feat_dim, alpha=config.alpha_uncertainty)
        
        self.proj_a = nn.Sequential(
            nn.Linear(config.audio_feat_dim, config.shared_dim),
            nn.LayerNorm(config.shared_dim),
            nn.Dropout(config.dropout)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(config.video_feat_dim, config.shared_dim),
            nn.LayerNorm(config.shared_dim),
            nn.Dropout(config.dropout)
        )
        
        # Multi-Scale 1D Temporal Convolutions
        self.conv_a1 = nn.Conv1d(config.shared_dim, config.shared_dim // 2, kernel_size=3, padding=1)
        self.conv_a2 = nn.Conv1d(config.shared_dim, config.shared_dim // 2, kernel_size=7, padding=3)
        self.conv_v1 = nn.Conv1d(config.shared_dim, config.shared_dim // 2, kernel_size=3, padding=1)
        self.conv_v2 = nn.Conv1d(config.shared_dim, config.shared_dim // 2, kernel_size=5, padding=2)
        
        # Bidirectional Cross Attention
        self.cross_attn = nn.MultiheadAttention(config.shared_dim, num_heads=4, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(config.shared_dim)
        
        # Attentive Temporal Pooling
        self.pool_net = nn.Sequential(
            nn.Linear(config.shared_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Linear(config.shared_dim, config.num_classes)

    def forward(self, audio_feats, video_feats):
        af = self.u_audio(audio_feats)
        vf = self.u_video(video_feats)
        
        ha = self.proj_a(af).transpose(1, 2)
        hv = self.proj_v(vf).transpose(1, 2)
        
        ha = F.gelu(torch.cat([self.conv_a1(ha), self.conv_a2(ha)], dim=1)).transpose(1, 2)
        hv = F.gelu(torch.cat([self.conv_v1(hv), self.conv_v2(hv)], dim=1)).transpose(1, 2)
        
        attn_out, _ = self.cross_attn(query=ha, key=hv, value=hv)
        fused_seq = self.norm(ha + attn_out)
        
        scores = self.pool_net(fused_seq)
        weights = F.softmax(scores, dim=1)
        z_av = (fused_seq * weights).sum(dim=1)
        
        logits_av = self.classifier(z_av)
        return z_av, logits_av

# ==================================================================================================
# 4. STAGE 2: CALIBRATED HIERARCHICAL FUSION (BEATING 73.62%)
# ==================================================================================================
class HierarchicalSOTAChampionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.text_backbone)
        self.shared_dim = config.shared_dim
        self.beta_shift = config.beta_shift
        
        # Stage 1: Non-Verbal AV Branch
        self.av_module = AudioVideoTemporalPyramid(config)
        
        # Stage 2: Text Adapter & Calibrated Gate
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
        
        # 2. Stage 1 Non-Verbal Representation
        z_av, logits_av = self.av_module(audio_feats, video_feats)
        
        # 3. Stage 2 Calibrated Residual Shift
        h_av = self.w_av_proj(z_av)
        t_norm = torch.sqrt(torch.sum(h_t ** 2, dim=-1, keepdim=True) + 1e-8)
        av_norm = torch.sqrt(torch.sum(h_av ** 2, dim=-1, keepdim=True) + 1e-8)
        threshold = (t_norm / av_norm) * self.beta_shift
        alpha = torch.clamp(threshold, max=1.0)
        
        gate = self.gate_mlp(torch.cat([h_t, h_av], dim=-1))
        
        shift = alpha * gate * h_av
        fused_rep = self.norm(h_t + shift)
        logits_fused = self.fused_classifier(self.dropout(fused_rep))
        
        return {
            'logits': logits_fused,
            'logits_text': logits_text,
            'logits_av': logits_av
        }

# ==================================================================================================
# 5. DATA LOADING & PADDING
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

class SOTAMIntRecDataset(Dataset):
    def __init__(self, tsv_path, audio_feats, video_feats, tokenizer):
        self.keys, self.labels, self.texts = [], [], []
        label_map = {name: idx for idx, name in enumerate(SOTAConfig.intent_labels)}
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
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=SOTAConfig.text_seq_len,
            return_tensors='pt'
        )
        return {
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'audio_feats': torch.tensor(self.audio_feats[key], dtype=torch.float32),
            'video_feats': torch.tensor(self.video_feats[key], dtype=torch.float32)
        }

# ==================================================================================================
# 6. EVALUATION FUNCTIONS
# ==================================================================================================
def evaluate_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            labels = b['label'].to(device)
            out = model(
                b['input_ids'].to(device), b['attention_mask'].to(device),
                b['audio_feats'].to(device), b['video_feats'].to(device)
            )
            all_logits.append(out['logits'].cpu())
            all_labels.extend(labels.cpu().numpy())
    return torch.cat(all_logits, dim=0).numpy(), np.array(all_labels)

def train_single_seed(seed, train_loader, dev_loader, test_loader, class_weights, device):
    print(f"\n" + "=" * 80)
    print(f">>> STARTING TRAINING FOR SEED {seed}")
    print("=" * 80)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    model = HierarchicalSOTAChampionModel(SOTAConfig).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=SOTAConfig.label_smoothing)
    
    # ----------------------------------------------------------------------------------------------
    # Phase 1: Pretrain Non-Verbal AV Branch (3 Epochs)
    # ----------------------------------------------------------------------------------------------
    av_opt = AdamW(model.av_module.parameters(), lr=SOTAConfig.lr_av_pretrain, weight_decay=0.01)
    for ep in range(1, SOTAConfig.epochs_av + 1):
        model.av_module.train()
        total_loss = 0.0
        for b in train_loader:
            af, vf, labels = b['audio_feats'].to(device), b['video_feats'].to(device), b['label'].to(device)
            av_opt.zero_grad()
            _, logits_av = model.av_module(af, vf)
            loss = criterion(logits_av, labels)
            loss.backward()
            av_opt.step()
            total_loss += loss.item()
            
    # ----------------------------------------------------------------------------------------------
    # Phase 2: Joint Calibrated Training (Early Stopping)
    # ----------------------------------------------------------------------------------------------
    bert_params = list(model.bert.parameters())
    fusion_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    optimizer = AdamW([
        {'params': bert_params, 'lr': SOTAConfig.lr_bert},
        {'params': fusion_params, 'lr': SOTAConfig.lr_fusion}
    ], weight_decay=SOTAConfig.weight_decay)
    
    total_steps = len(train_loader) * SOTAConfig.epochs_joint
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    best_dev_acc = 0.0
    best_weights = None
    patience = 0
    
    for ep in range(1, SOTAConfig.epochs_joint + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            labels = b['label'].to(device)
            out = model(b['input_ids'].to(device), b['attention_mask'].to(device),
                        b['audio_feats'].to(device), b['video_feats'].to(device))
            loss = criterion(out['logits'], labels) + 0.3 * criterion(out['logits_text'], labels) + 0.2 * criterion(out['logits_av'], labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        dev_logits, dev_labels = evaluate_logits(model, dev_loader, device)
        dev_acc = accuracy_score(dev_labels, np.argmax(dev_logits, axis=-1)) * 100.0
        dev_wf1 = f1_score(dev_labels, np.argmax(dev_logits, axis=-1), average='weighted') * 100.0
        print(f"Seed {seed} | Epoch {ep:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_acc:.2f}% | Dev WF1: {dev_wf1:.2f}%")
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
            print(f"    --> [BEST SEED {seed}] Dev Acc: {best_dev_acc:.2f}%")
        else:
            patience += 1
            if patience >= SOTAConfig.patience:
                print(f"[*] Early stopping triggered at Epoch {ep}!")
                break
                
    model.load_state_dict(best_weights)
    test_logits, test_labels = evaluate_logits(model, test_loader, device)
    test_acc = accuracy_score(test_labels, np.argmax(test_logits, axis=-1)) * 100.0
    test_wf1 = f1_score(test_labels, np.argmax(test_logits, axis=-1), average='weighted') * 100.0
    print(f"\n[+] Seed {seed} Test Accuracy : {test_acc:.2f}% | Weighted F1: {test_wf1:.2f}%")
    return test_logits, test_labels, test_acc, test_wf1

# ==================================================================================================
# 7. MAIN RUNNER (SEEDS 0, 1, 2 + SOFTMAX ENSEMBLE)
# ==================================================================================================
def run_sota_champion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target Device: {device}")
    
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
    audio_feats = pad_features(audio_raw, keys, SOTAConfig.audio_seq_len)
    video_feats = pad_features(video_raw, keys, SOTAConfig.video_seq_len)
    
    tokenizer = BertTokenizer.from_pretrained(SOTAConfig.text_backbone, do_lower_case=True)
    
    train_ds = SOTAMIntRecDataset(data_dir / "train.tsv", audio_feats, video_feats, tokenizer)
    dev_ds = SOTAMIntRecDataset(data_dir / "dev.tsv", audio_feats, video_feats, tokenizer)
    test_ds = SOTAMIntRecDataset(data_dir / "test.tsv", audio_feats, video_feats, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=SOTAConfig.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=SOTAConfig.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=SOTAConfig.eval_batch_size, shuffle=False)
    
    class_weights = torch.ones(SOTAConfig.num_classes, dtype=torch.float32)
    for c in ['Joke', 'Taunt', 'Flaunt', 'Criticize']:
        class_weights[SOTAConfig.intent_labels.index(c)] = SOTAConfig.conflict_weight
    class_weights = class_weights.to(device)
    
    all_test_logits = []
    seed_accs, seed_wf1s = [], []
    
    # Run across seeds [0, 1, 2]
    for seed in SOTAConfig.seeds:
        test_logits, test_labels, t_acc, t_wf1 = train_single_seed(
            seed, train_loader, dev_loader, test_loader, class_weights, device
        )
        all_test_logits.append(test_logits)
        seed_accs.append(t_acc)
        seed_wf1s.append(t_wf1)
        
    # ----------------------------------------------------------------------------------------------
    # SOFTMAX ENSEMBLE ARBITRATION (CHAMPION ACCURACY)
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> FINAL MULTI-SEED SUMMARY & ENSEMBLE ARBITRATION")
    print("=" * 80)
    for s, acc, wf1 in zip(SOTAConfig.seeds, seed_accs, seed_wf1s):
        print(f"Seed {s:<3} : Test Accuracy = {acc:.2f}% | Weighted F1 = {wf1:.2f}%")
        
    mean_acc = np.mean(seed_accs)
    mean_wf1 = np.mean(seed_wf1s)
    print("-" * 80)
    print(f"3-Seed Mean Performance : Accuracy = {mean_acc:.2f}% | Weighted F1 = {mean_wf1:.2f}%")
    
    # Softmax probabilities averaged across all 3 seeds
    softmax_probs = [F.softmax(torch.tensor(l), dim=-1).numpy() for l in all_test_logits]
    ensemble_prob = np.mean(softmax_probs, axis=0)
    ensemble_preds = np.argmax(ensemble_prob, axis=-1)
    
    ens_acc = accuracy_score(test_labels, ensemble_preds) * 100.0
    ens_wf1 = f1_score(test_labels, ensemble_preds, average='weighted') * 100.0
    ens_mf1 = f1_score(test_labels, ensemble_preds, average='macro') * 100.0
    
    print("=" * 80)
    print(f"🏆 ENSEMBLE TEST ACCURACY   : {ens_acc:.2f}% (Target: >73.62%)")
    print(f"🏆 ENSEMBLE WEIGHTED F1     : {ens_wf1:.2f}% (Target: >73.31%)")
    print(f"🏆 ENSEMBLE MACRO F1        : {ens_mf1:.2f}%")
    print("=" * 80)
    
    paper_acc = 73.62
    if ens_acc > paper_acc:
        print(f"\n🎉 CONGRATULATIONS! YOU OFFICIALLY SURPASSED THE AAAI 2024 PAPER BENCHMARK ({paper_acc}%) BY +{ens_acc - paper_acc:.2f}%!")
    else:
        print(f"\n[+] Highest Single Seed ACC : {max(seed_accs):.2f}%")

if __name__ == "__main__":
    run_sota_champion()
