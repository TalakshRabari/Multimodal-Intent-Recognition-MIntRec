"""
====================================================================================================
    HIERARCHICAL MULTIMODAL INTENT RECOGNITION V2: PERFORMANCE OPTIMIZATION SUITE
====================================================================================================
Project:       Multimodal Intent Recognition (Major Project)
Authors:       Talaksh Rabari & Swayam Patel
Target:        Improve performance from 70.1% baseline toward 75%+ SOTA by targeting the
               bottlenecks revealed in the initial experiment plots:
               1. Fix 26.7% Mutual Failure on Minority Classes (Focal / Class-Balanced Loss)
               2. Upgrade Audio-Video Branch from 24.7% -> 35%+ (Temporal 1D-CNN Multi-Scale)
               3. Dynamic Entropy & Margin-Based Sequential Gating (Replace static tau)
               4. Full 10-Epoch Training with Cosine Warmup & Cross-Modal InfoNCE Alignment
====================================================================================================
"""

import __main__
if not hasattr(__main__, '__file__'):
    __main__.__file__ = '/kaggle/working/hierarchical_multimodal_v2.py'

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==================================================================================================
# 1. ENHANCED CONFIGURATION & CLASS IMBALANCE WEIGHTS
# ==================================================================================================
class ConfigV2:
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
    dropout = 0.2
    
    # Training Parameters
    num_classes = 20
    train_batch_size = 16
    eval_batch_size = 16
    lr_bert = 2e-5
    lr_fusion = 1e-4
    lr_av = 2e-4
    weight_decay = 0.01
    num_epochs = 10
    seed = 42
    
    # Loss Configuration
    loss_type = "focal_balanced"       # 'focal_balanced', 'labelsmooth', or 'ce'
    focal_gamma = 1.5
    label_smoothing = 0.05
    lambda_infonce = 0.10              # Cross-modal consistency loss weight
    
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help' 
    ]
    
    # Exact training set class frequencies (derived from train.tsv)
    train_class_counts = [
        172, 127, 82, 74, 70, 57, 35, 38, 32, 31, 31, 52, 170, 73, 66, 63, 51, 43, 36, 31
    ]

def get_class_weights(counts, beta=0.99, device="cpu"):
    """Effective Number of Samples Class Weights (Cui et al., CVPR 2019)."""
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)

# ==================================================================================================
# 2. ADVANCED LOSS FUNCTIONS: FOCAL & CLASS-BALANCED
# ==================================================================================================
class ClassBalancedFocalLoss(nn.Module):
    """
    Addresses the 0% win-rate bottleneck on minority classes (Joke, Taunt, Oppose, Flaunt).
    Combines inverse effective frequency weighting with focal modulation (1 - p_t)^gamma.
    """
    def __init__(self, class_weights=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)
        
        # Smooth one-hot encoding
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, fill_value=self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow(1.0 - p_t + 1e-6, self.gamma)
        
        loss = -(smooth_targets * log_p).sum(dim=-1) * focal_weight
        
        if self.class_weights is not None:
            c_weights = self.class_weights.gather(0, targets)
            loss = loss * c_weights
            
        return loss.mean()

class CrossModalInfoNCELoss(nn.Module):
    """Aligns Audio-Video representation with Text representation to enforce cross-modal semantic agreement."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_t, z_av):
        z_t_norm = F.normalize(z_t, dim=-1)
        z_av_norm = F.normalize(z_av, dim=-1)
        
        # Similarity matrix: (B, B)
        sim = torch.matmul(z_t_norm, z_av_norm.T) / self.temperature
        labels = torch.arange(z_t.size(0), device=z_t.device)
        
        loss_t2av = F.cross_entropy(sim, labels)
        loss_av2t = F.cross_entropy(sim.T, labels)
        return (loss_t2av + loss_av2t) / 2.0

# ==================================================================================================
# 3. UPGRADED AUDIO-VIDEO BRANCH: TEMPORAL 1D-CNN + ATTENTIVE POOLING
# ==================================================================================================
class Temporal1DConvPyramid(nn.Module):
    """
    Captures multi-scale temporal dynamics:
    - Kernel 3: Micro-expressions (frown, smirk, blink) and phoneme-level pitch transitions.
    - Kernel 7: Clause-level prosody and sustained posture.
    """
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv3 = nn.Conv1d(out_dim, out_dim // 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(out_dim, out_dim // 2, kernel_size=7, padding=3)
        self.norm = nn.GroupNorm(8, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, L, D) -> (B, D, L)
        h = self.proj(x).transpose(1, 2)
        c3 = self.conv3(h)
        c7 = self.conv7(h)
        out = F.gelu(self.norm(torch.cat([c3, c7], dim=1)))
        out = self.dropout(out).transpose(1, 2)  # (B, L, out_dim)
        return out

class AttentiveTemporalPooling(nn.Module):
    """Replaces naive mean pooling with learned attention to focus on peak emotional moments."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    def forward(self, x):
        # x: (B, L, D)
        scores = self.attn(x)  # (B, L, 1)
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights).sum(dim=1)  # (B, D)
        return pooled

class EnhancedAudioVideoFusion(nn.Module):
    """
    Upgraded Stage 1 Non-Verbal Fusion with 1D-CNN Temporal Pyramids & Attentive Pooling.
    """
    def __init__(self, audio_dim=768, video_dim=1024, shared_dim=256, nheads=4, dropout=0.2):
        super().__init__()
        self.audio_pyramid = Temporal1DConvPyramid(audio_dim, shared_dim, dropout=dropout)
        self.video_pyramid = Temporal1DConvPyramid(video_dim, shared_dim, dropout=dropout)
        
        # Bidirectional Cross-Modal Attention
        self.cross_a2v = nn.MultiheadAttention(shared_dim, nheads, dropout=dropout, batch_first=True)
        self.cross_v2a = nn.MultiheadAttention(shared_dim, nheads, dropout=dropout, batch_first=True)
        
        self.norm_a = nn.LayerNorm(shared_dim)
        self.norm_v = nn.LayerNorm(shared_dim)
        
        self.pool_a = AttentiveTemporalPooling(shared_dim)
        self.pool_v = AttentiveTemporalPooling(shared_dim)
        
        # Adaptive Multi-Scale Fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid()
        )
        self.final_norm = nn.LayerNorm(shared_dim)

    def forward(self, audio_feats, video_feats, w_audio=0.5, w_video=0.5):
        h_a = self.audio_pyramid(audio_feats)  # (B, L_a, shared_dim)
        h_v = self.video_pyramid(video_feats)  # (B, L_v, shared_dim)
        
        # Cross Attention
        attn_a, _ = self.cross_a2v(h_a, h_v, h_v)
        h_a = self.norm_a(h_a + attn_a)
        
        attn_v, _ = self.cross_v2a(h_v, h_a, h_a)
        h_v = self.norm_v(h_v + attn_v)
        
        p_a = self.pool_a(h_a)  # (B, shared_dim)
        p_v = self.pool_v(h_v)  # (B, shared_dim)
        
        w_sum = w_audio + w_video + 1e-9
        weighted_base = (w_audio / w_sum) * p_a + (w_video / w_sum) * p_v
        
        gate = self.fusion_gate(torch.cat([p_a, p_v], dim=-1))
        z_av = self.final_norm(weighted_base + gate * (p_a * p_v))
        return z_av, p_a, p_v

# ==================================================================================================
# 4. FULL HIERARCHICAL MODEL V2 WITH DYNAMIC ENTROPY GATING
# ==================================================================================================
class HierarchicalIntentModelV2(nn.Module):
    def __init__(self, text_backbone="bert-base-uncased", shared_dim=256, num_classes=20, dropout=0.2):
        super().__init__()
        self.bert = BertModel.from_pretrained(text_backbone)
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        
        # Upgraded Stage 1: Non-Verbal AV Fusion
        self.av_fusion = EnhancedAudioVideoFusion(
            audio_dim=768, video_dim=1024, shared_dim=shared_dim, nheads=4, dropout=dropout
        )
        
        # Stage 2: Text Adapter
        self.text_proj = nn.Sequential(
            nn.Linear(768, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Adaptive Bimodal Gate (MAG)
        self.mag_gate = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.Sigmoid()
        )
        self.mag_norm = nn.LayerNorm(shared_dim)
        
        # Classifiers
        self.text_classifier = nn.Linear(shared_dim, num_classes)
        self.av_classifier = nn.Linear(shared_dim, num_classes)
        
        self.fused_classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes)
        )
        
        # Dynamic Confidence & Entropy Router
        self.router_net = nn.Sequential(
            nn.Linear(num_classes * 2 + 2, shared_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask, audio_feats, video_feats,
                w_text=1.0, w_audio=0.5, w_video=0.5, use_dynamic_gating=True):
        # 1. Text Encoding
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]
        h_t = self.text_proj(cls_token)
        logits_text = self.text_classifier(h_t)
        
        # 2. Stage 1 Non-Verbal Fusion
        z_av, p_a, p_v = self.av_fusion(audio_feats, video_feats, w_audio, w_video)
        logits_av = self.av_classifier(z_av)
        
        # 3. Stage 2 Hierarchical Fusion
        cat_t_av = torch.cat([h_t, z_av], dim=-1)
        gate = self.mag_gate(cat_t_av)
        fused_rep = self.mag_norm(w_text * h_t + (1.0 - w_text * 0.5) * (gate * z_av))
        logits_fused = self.fused_classifier(fused_rep)
        
        # 4. Softmax & Information-Theoretic Uncertainty Metrics
        prob_text = F.softmax(logits_text, dim=-1)
        conf_text, pred_text = torch.max(prob_text, dim=-1)
        
        # Prediction Entropy: H(p) = -sum(p * log p)
        entropy_text = -(prob_text * torch.log(prob_text + 1e-9)).sum(dim=-1, keepdim=True)
        # Margin: Top1 - Top2
        top2 = torch.topk(prob_text, k=2, dim=-1).values
        margin_text = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)
        
        # 5. Dynamic Adaptive Routing
        if use_dynamic_gating:
            routing_inputs = torch.cat([logits_text, logits_av, entropy_text, margin_text], dim=-1)
            residual = self.router_net(routing_inputs)
            # Modulate fused logits: when entropy is high or margin is low, non-verbal residual takes charge
            logits_output = logits_fused + 0.3 * residual
        else:
            logits_output = logits_fused
            
        return {
            'logits': logits_output,
            'logits_text': logits_text,
            'logits_av': logits_av,
            'z_av': z_av,
            'h_t': h_t,
            'conf_text': conf_text,
            'entropy_text': entropy_text.squeeze(-1)
        }

# ==================================================================================================
# 5. DATASET LOADING & PADDING UTILITY
# ==================================================================================================
def pad_features(raw_feats, keys, max_len):
    padded_dict = {}
    for k in keys:
        if k in raw_feats:
            f = raw_feats[k]
            if len(f.shape) == 3:
                f = f.squeeze(1)
            cur_len, feat_dim = f.shape[0], f.shape[1]
            if cur_len >= max_len:
                padded = f[:max_len, :]
            else:
                padded = np.pad(f, ((0, max_len - cur_len), (0, 0)), mode='constant')
            padded_dict[k] = padded
        else:
            feat_dim = 768 if 'audio' in str(k).lower() else 1024
            padded_dict[k] = np.zeros((max_len, feat_dim), dtype=np.float32)
    return padded_dict

class MIntRecDatasetV2(Dataset):
    def __init__(self, tsv_path, audio_feats, video_feats, tokenizer, max_text_len=30):
        self.rows, self.keys, self.labels, self.texts = [], [], [], []
        label_map = {name: idx for idx, name in enumerate(ConfigV2.intent_labels)}
        
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for r in reader:
                if len(r) >= 5:
                    self.rows.append(r)
                    self.keys.append(f"{r[0]}_{r[1]}_{r[2]}")
                    self.texts.append(r[3])
                    self.labels.append(label_map[r[4]])
                    
        self.audio_feats = audio_feats
        self.video_feats = video_feats
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        key = self.keys[idx]
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
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
# 6. TRAINING & EVALUATION LOOP
# ==================================================================================================
def evaluate(model, loader, device):
    model.eval()
    preds, preds_t, preds_av, labels_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch['label'].to(device)
            inp = batch['input_ids'].to(device)
            att = batch['attention_mask'].to(device)
            af = batch['audio_feats'].to(device)
            vf = batch['video_feats'].to(device)
            
            out = model(inp, att, af, vf)
            preds.extend(torch.argmax(out['logits'], dim=-1).cpu().numpy())
            preds_t.extend(torch.argmax(out['logits_text'], dim=-1).cpu().numpy())
            preds_av.extend(torch.argmax(out['logits_av'], dim=-1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            
    y_true = np.array(labels_all)
    y_pred = np.array(preds)
    acc = accuracy_score(y_true, y_pred) * 100.0
    wf1 = f1_score(y_true, y_pred, average='weighted') * 100.0
    mf1 = f1_score(y_true, y_pred, average='macro') * 100.0
    
    acc_t = accuracy_score(y_true, np.array(preds_t)) * 100.0
    acc_av = accuracy_score(y_true, np.array(preds_av)) * 100.0
    
    return {
        'acc': acc, 'wf1': wf1, 'mf1': mf1,
        'acc_text': acc_t, 'acc_av': acc_av,
        'preds': y_pred, 'labels': y_true
    }

def train_enhanced_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_bert", type=float, default=2e-5)
    parser.add_argument("--lr_fusion", type=float, default=1e-4)
    args, _ = parser.parse_known_args()
    
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
    audio_feats = pad_features(audio_raw, keys, ConfigV2.audio_seq_len)
    video_feats = pad_features(video_raw, keys, ConfigV2.video_seq_len)
    
    tokenizer = BertTokenizer.from_pretrained(ConfigV2.text_backbone, do_lower_case=True)
    
    train_ds = MIntRecDatasetV2(data_dir / "train.tsv", audio_feats, video_feats, tokenizer)
    dev_ds = MIntRecDatasetV2(data_dir / "dev.tsv", audio_feats, video_feats, tokenizer)
    test_ds = MIntRecDatasetV2(data_dir / "test.tsv", audio_feats, video_feats, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    model = HierarchicalIntentModelV2(
        text_backbone=ConfigV2.text_backbone,
        shared_dim=ConfigV2.shared_dim,
        num_classes=ConfigV2.num_classes,
        dropout=ConfigV2.dropout
    ).to(device)
    
    class_weights = get_class_weights(ConfigV2.train_class_counts, beta=0.99, device=device)
    criterion_focal = ClassBalancedFocalLoss(class_weights=class_weights, gamma=ConfigV2.focal_gamma)
    criterion_infonce = CrossModalInfoNCELoss(temperature=0.07)
    
    bert_params = list(model.bert.parameters())
    fusion_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': args.lr_bert},
        {'params': fusion_params, 'lr': args.lr_fusion}
    ], weight_decay=ConfigV2.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    best_dev_acc = 0.0
    best_model_weights = None
    
    print("\n" + "=" * 80)
    print(">>> STARTING V2 ENHANCED TRAINING (Focal Loss + Temporal 1D-CNN + InfoNCE)")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels = batch['label'].to(device)
            inp = batch['input_ids'].to(device)
            att = batch['attention_mask'].to(device)
            af = batch['audio_feats'].to(device)
            vf = batch['video_feats'].to(device)
            
            optimizer.zero_grad()
            out = model(inp, att, af, vf)
            
            loss_fused = criterion_focal(out['logits'], labels)
            loss_t = criterion_focal(out['logits_text'], labels)
            loss_av = criterion_focal(out['logits_av'], labels)
            loss_nce = criterion_infonce(out['h_t'], out['z_av'])
            
            loss = loss_fused + 0.3 * loss_t + 0.3 * loss_av + ConfigV2.lambda_infonce * loss_nce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        dev_res = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_res['acc']:.2f}% | WF1: {dev_res['wf1']:.2f}% | AV Acc: {dev_res['acc_av']:.2f}%")
        
        if dev_res['acc'] > best_dev_acc:
            best_dev_acc = dev_res['acc']
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"    --> [NEW BEST] Saved checkpoint with Dev Acc: {best_dev_acc:.2f}%")
            
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    print("\n" + "=" * 80)
    print(">>> FINAL TEST EVALUATION (V2 ENHANCED MODEL)")
    print("=" * 80)
    test_res = evaluate(model, test_loader, device)
    print(f"[+] Final Test Accuracy : {test_res['acc']:.2f}%")
    print(f"[+] Final Weighted F1   : {test_res['wf1']:.2f}%")
    print(f"[+] Final Macro F1      : {test_res['mf1']:.2f}")
    print(f"[+] Text-Only Accuracy  : {test_res['acc_text']:.2f}%")
    print(f"[+] Audio-Video Accuracy: {test_res['acc_av']:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    train_enhanced_model()
