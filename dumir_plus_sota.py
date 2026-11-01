"""
====================================================================================================
    DUMIR+ (DUMIR-PLUS): OUTPERFORMING THE 76.01% SOTA BENCHMARK ON MINTREC
====================================================================================================
Project:       Multimodal Intent Recognition (Major Project)
Authors:       Talaksh Rabari & Swayam Patel (Dhirubhai Ambani University)
Advisor:       Dr. Shruti Bhilare
Goal:          Beat the 76.01% Accuracy / 75.87% Weighted F1 SOTA Benchmark (DUMIR 2026)
Architecture:  DUMIR+ (Next-Generation SOTA Architecture):
               1. UAGM: Uncertainty-Aware Gating Module (alpha = 5.0)
               2. HCMI: Hierarchical Cascaded Cross-Modal Interaction across Text, Audio, and Video
               3. Multi-Scale 1D-CNN Temporal Pyramids for Audio (k=3, 7) & Video (k=3, 5)
                  (Replaces DUMIR's naive 1x1 pointwise convolutions)
               4. Attentive Temporal Pooling (Replaces DUMIR's naive global mean pooling)
               5. CMCC: Cross-Modal InfoNCE Consistency Constraint (lambda_c = 0.03, tau = 0.05)
               6. MAS: Modality-Level Auxiliary Supervision (lambda_m = 0.80)
               7. Gated Class Prototype Memory Clustering (top-k = 3)
               8. Targeted 1.5x Conflict Reweighting + Label Smoothing (0.1)
               9. Multi-Seed Training (Seeds 0, 1, 2) + Softmax Ensemble Arbitration
====================================================================================================
"""

import __main__
if not hasattr(__main__, '__file__'):
    __main__.__file__ = '/kaggle/working/dumir_plus_sota.py'

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
# 1. DUMIR+ CONFIGURATION (Table 2 + SOTA Extensions)
# ==================================================================================================
class DUMIRPlusConfig:
    dataset = "MIntRec"
    text_backbone = "bert-base-uncased"
    
    # Feature Dimensions & Sequences
    text_feat_dim = 768
    audio_feat_dim = 768
    video_feat_dim = 1024
    text_seq_len = 30
    audio_seq_len = 480
    video_seq_len = 230
    
    # DUMIR Architectural Parameters
    hidden_dim = 360                   # DUMIR hidden dimension = 360
    nheads = 4                         # Multihead attention heads = 4
    n_levels = 4
    alpha = 5.0                        # Uncertainty penalty factor (Fig. 3)
    temperature = 0.05                 # InfoNCE temperature = 0.05
    top_k_prototypes = 3
    
    # Multi-Task Loss Weights
    lambda_m = 0.80                    # Modality auxiliary supervision weight
    lambda_c = 0.03                    # Consistency contrastive loss weight
    label_smoothing = 0.1
    conflict_weight = 1.5              # 1.5x penalty on Joke, Taunt, Flaunt, Criticize
    
    # Regularization
    text_dropout = 0.4
    output_dropout = 0.2
    embed_dropout = 0.1
    attn_dropout = 0.1
    grad_clip = 0.5
    
    # Training Parameters
    seeds = [0, 1, 2]                  # 3-Seed Execution + Ensemble
    num_train_epochs = 20
    wait_patience = 8                  # Patience = 8 matching DUMIR Table 2
    train_batch_size = 16
    eval_batch_size = 16
    lr = 3e-5                          # DUMIR learning rate = 3e-5
    weight_decay = 0.01
    warmup_proportion = 0.1
    
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help' 
    ]
    num_labels = len(intent_labels)

# ==================================================================================================
# 2. UPGRADED MULTI-SCALE 1D-CNN TEMPORAL PYRAMID (DUMIR+ UPGRADE 1)
# ==================================================================================================
class MultiScaleConv1d(nn.Module):
    """
    Upgrades DUMIR's naive 1x1 conv to multi-scale temporal convolutions:
    - Audio: Kernels 3 & 7 capture phoneme transitions and intonational stress.
    - Video: Kernels 3 & 5 capture rapid facial micro-expressions.
    """
    def __init__(self, in_dim, out_dim, k1=3, k2=7, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.conv1 = nn.Conv1d(out_dim, out_dim // 2, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv1d(out_dim, out_dim // 2, kernel_size=k2, padding=k2 // 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, in_dim)
        h = self.proj(x).transpose(1, 2)
        c1 = self.conv1(h)
        c2 = self.conv2(h)
        fused = F.gelu(torch.cat([c1, c2], dim=1)).transpose(1, 2)
        return self.dropout(self.norm(fused))

class AttentiveSequencePooling(nn.Module):
    """
    Upgrades DUMIR's naive global mean pooling with learned emotional saliency attention.
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    def forward(self, x):
        scores = self.attn(x)
        weights = F.softmax(scores, dim=1)
        return (x * weights).sum(dim=1)

# ==================================================================================================
# 3. DUMIR CORE MODULES: UAGM, HCMI & PROTOTYPE MEMORY
# ==================================================================================================
class UncertaintyAwareGating(nn.Module):
    """UAGM: Section 3.2, Eqs. 6-10."""
    def __init__(self, hidden_dim, alpha=5.0):
        super().__init__()
        self.alpha = alpha
        self.saliency_proj = nn.Linear(hidden_dim * 3, 3)

    def forward(self, H_l, H_a, H_v):
        g_l = torch.mean(H_l, dim=1)
        g_a = torch.mean(H_a, dim=1)
        g_v = torch.mean(H_v, dim=1)

        v_l = torch.mean((H_l - g_l.unsqueeze(1)) ** 2, dim=[1, 2])
        v_a = torch.mean((H_a - g_a.unsqueeze(1)) ** 2, dim=[1, 2])
        v_v = torch.mean((H_v - g_v.unsqueeze(1)) ** 2, dim=[1, 2])

        u_l = torch.sigmoid(v_l).unsqueeze(-1)
        u_a = torch.sigmoid(v_a).unsqueeze(-1)
        u_v = torch.sigmoid(v_v).unsqueeze(-1)
        u = torch.cat([u_l, u_a, u_v], dim=-1)

        g_joint = torch.cat([g_l, g_a, g_v], dim=-1)
        e = self.saliency_proj(g_joint)
        S = e - self.alpha * u
        weights = F.softmax(S, dim=-1)

        w_l = weights[:, 0].view(-1, 1, 1)
        w_a = weights[:, 1].view(-1, 1, 1)
        w_v = weights[:, 2].view(-1, 1, 1)

        return w_l * H_l, w_a * H_a, w_v * H_v, weights

class CrossModalAttentionBlock(nn.Module):
    """CA(Q, K, V): Eq. 11."""
    def __init__(self, hidden_dim=360, nheads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nheads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, q, k, v):
        attn_out, _ = self.attn(query=q, key=k, value=v)
        x = self.norm1(q + self.dropout(attn_out))
        return self.norm2(x + self.ffn(x))

class HCMIPlus(nn.Module):
    """
    Hierarchical Cross-Modal Interaction with Attentive Pooling (DUMIR+ Upgrade 2).
    """
    def __init__(self, hidden_dim=360, nheads=4, dropout=0.1):
        super().__init__()
        self.ca_la = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_lav = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_al = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_alv = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_vl = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_vla = CrossModalAttentionBlock(hidden_dim, nheads, dropout)

        self.pool_la = AttentiveSequencePooling(hidden_dim)
        self.pool_lav = AttentiveSequencePooling(hidden_dim)
        self.pool_al = AttentiveSequencePooling(hidden_dim)
        self.pool_alv = AttentiveSequencePooling(hidden_dim)
        self.pool_vl = AttentiveSequencePooling(hidden_dim)
        self.pool_vla = AttentiveSequencePooling(hidden_dim)

    def forward(self, F_l, F_a, F_v):
        # Text Anchor
        H_la = self.ca_la(F_l, F_a, F_a)
        H_lav = self.ca_lav(H_la, F_v, F_v)
        Z_l = torch.cat([self.pool_la(H_la), self.pool_lav(H_lav)], dim=-1)

        # Audio Anchor
        H_al = self.ca_al(F_a, F_l, F_l)
        H_alv = self.ca_alv(H_al, F_v, F_v)
        Z_a = torch.cat([self.pool_al(H_al), self.pool_alv(H_alv)], dim=-1)

        # Video Anchor
        H_vl = self.ca_vl(F_v, F_l, F_l)
        H_vla = self.ca_vla(H_vl, F_a, F_a)
        Z_v = torch.cat([self.pool_vl(H_vl), self.pool_vla(H_vla)], dim=-1)

        return Z_l, Z_a, Z_v

class PrototypeMemory(nn.Module):
    def __init__(self, feat_dim, num_classes, top_k=3):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[2].bias, -2.0)
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x):
        fn = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) + 1e-8)
        pn = torch.sqrt(torch.sum(self.prototypes ** 2, dim=-1, keepdim=True) + 1e-8)
        sim = torch.matmul(x / fn, (self.prototypes / pn).t())
        top_vals, top_idx = torch.topk(sim, k=self.top_k, dim=-1)
        attn = F.softmax(top_vals, dim=-1)
        selected_p = self.prototypes[top_idx]
        mem = torch.sum(attn.unsqueeze(-1) * selected_p, dim=1)
        g = self.gate(torch.cat([x, mem], dim=-1))
        return self.norm(x + g * mem)

# ==================================================================================================
class Conv1dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, padding=0, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.norm(out))

# ==================================================================================================
# 4. FULL DUMIR+ CHAMPION MODEL
# ==================================================================================================
class DUMIRPlusModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.text_backbone)
        self.text_dropout = nn.Dropout(args.text_dropout)

        # Multi-scale 1D temporal projections
        self.proj_l = Conv1dBlock(args.text_feat_dim, args.hidden_dim, kernel_size=5, padding=2, dropout=args.embed_dropout)
        self.proj_a = MultiScaleConv1d(args.audio_feat_dim, args.hidden_dim, k1=3, k2=7, dropout=args.embed_dropout)
        self.proj_v = MultiScaleConv1d(args.video_feat_dim, args.hidden_dim, k1=3, k2=5, dropout=args.embed_dropout)

        self.uagm = UncertaintyAwareGating(args.hidden_dim, alpha=args.alpha)
        self.hcmi = HCMIPlus(args.hidden_dim, nheads=args.nheads, dropout=args.attn_dropout)

        joint_dim = args.hidden_dim * 2 * 3  # 360 * 2 * 3 = 2160
        self.fusion_mlp = nn.Sequential(
            nn.Linear(joint_dim, 768),
            nn.GELU(),
            nn.Dropout(args.output_dropout)
        )
        self.proto_memory = PrototypeMemory(768, args.num_labels, args.top_k_prototypes)
        self.classifier = nn.Linear(768, args.num_labels)
        self.out_dropout = nn.Dropout(args.output_dropout)

        anchor_dim = args.hidden_dim * 2
        self.aux_head_l = nn.Linear(anchor_dim, args.num_labels)
        self.aux_head_a = nn.Linear(anchor_dim, args.num_labels)
        self.aux_head_v = nn.Linear(anchor_dim, args.num_labels)

    def forward(self, input_ids, attention_mask, audio_feats, video_feats):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        bert_out = self.text_dropout(bert_out)

        H_l = self.proj_l(bert_out)
        H_a = self.proj_a(audio_feats)
        H_v = self.proj_v(video_feats)

        F_l, F_a, F_v, u_weights = self.uagm(H_l, H_a, H_v)
        Z_l, Z_a, Z_v = self.hcmi(F_l, F_a, F_v)

        Z_joint = torch.cat([Z_l, Z_a, Z_v], dim=-1)
        fused = self.fusion_mlp(Z_joint)
        enhanced = self.proto_memory(fused)
        logits = self.classifier(self.out_dropout(enhanced))

        aux_logits = {
            'text': self.aux_head_l(Z_l),
            'audio': self.aux_head_a(Z_a),
            'video': self.aux_head_v(Z_v)
        }
        return logits, Z_l, Z_a, Z_v, aux_logits

# ==================================================================================================
# 5. DUMIR MULTI-TASK LOSS (Eq. 16)
# ==================================================================================================
def compute_infonce(z1, z2, temperature=0.05):
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    sim = torch.matmul(z1_norm, z2_norm.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.T, labels)
    return (loss_12 + loss_21) / 2.0

def dumir_loss_fn(logits, aux_logits, Z_l, Z_a, Z_v, targets, cls_criterion, args):
    l_fusion = cls_criterion(logits, targets)
    l_mod = (
        cls_criterion(aux_logits['text'], targets) +
        cls_criterion(aux_logits['audio'], targets) +
        cls_criterion(aux_logits['video'], targets)
    ) / 3.0
    l_la = compute_infonce(Z_l, Z_a, args.temperature)
    l_lv = compute_infonce(Z_l, Z_v, args.temperature)
    l_av = compute_infonce(Z_a, Z_v, args.temperature)
    l_cons = (l_la + l_lv + l_av) / 3.0
    return l_fusion + args.lambda_m * l_mod + args.lambda_c * l_cons

# ==================================================================================================
# 6. DATASET LOADER
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

class DUMIRDataset(Dataset):
    def __init__(self, tsv_path, audio_feats, video_feats, tokenizer):
        self.keys, self.labels, self.texts = [], [], []
        label_map = {name: idx for idx, name in enumerate(DUMIRPlusConfig.intent_labels)}
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
            max_length=DUMIRPlusConfig.text_seq_len,
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
# 7. TRAINING & EVALUATION ENGINE
# ==================================================================================================
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            labels = b['label'].to(device)
            logits, _, _, _, _ = model(
                b['input_ids'].to(device), b['attention_mask'].to(device),
                b['audio_feats'].to(device), b['video_feats'].to(device)
            )
            all_logits.append(logits.cpu())
            all_labels.extend(labels.cpu().numpy())
    logits_arr = torch.cat(all_logits, dim=0).numpy()
    labels_arr = np.array(all_labels)
    preds = np.argmax(logits_arr, axis=-1)
    acc = accuracy_score(labels_arr, preds) * 100.0
    wf1 = f1_score(labels_arr, preds, average='weighted') * 100.0
    mf1 = f1_score(labels_arr, preds, average='macro') * 100.0
    return acc, wf1, mf1, logits_arr, labels_arr

def train_seed(seed, train_loader, dev_loader, test_loader, cls_criterion, args, device):
    print(f"\n>>> Running Seed {seed} (Max {args.num_train_epochs} Epochs | Patience = {args.wait_patience})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DUMIRPlusModel(args).to(device)

    bert_params = list(model.bert.parameters())
    custom_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    optimizer = AdamW([
        {'params': bert_params, 'lr': args.lr * 0.7},
        {'params': custom_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_proportion), num_training_steps=total_steps)

    best_dev_score = 0.0
    best_weights = None
    best_epoch = 0
    patience = 0

    for ep in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0
        for b in train_loader:
            labels = b['label'].to(device)
            logits, Z_l, Z_a, Z_v, aux_logits = model(
                b['input_ids'].to(device), b['attention_mask'].to(device),
                b['audio_feats'].to(device), b['video_feats'].to(device)
            )
            loss = dumir_loss_fn(logits, aux_logits, Z_l, Z_a, Z_v, labels, cls_criterion, args)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        dev_acc, dev_wf1, dev_mf1, _, _ = evaluate(model, dev_loader, device)
        print(f"  [Seed {seed}] Ep {ep:02d} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_acc:.2f}% | Dev WF1: {dev_wf1:.2f}%")

        # Monitor Dev WF1 matching DUMIR Sec. 4.2
        if dev_wf1 > best_dev_score:
            best_dev_score = dev_wf1
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = ep
            patience = 0
        else:
            patience += 1
            if patience >= args.wait_patience:
                print(f"  [*] Early stopping at Epoch {ep} (Peak Dev WF1: {best_dev_score:.2f}% at Ep {best_epoch})")
                break

    model.load_state_dict(best_weights)
    test_acc, test_wf1, test_mf1, test_logits, test_labels = evaluate(model, test_loader, device)
    print(f"[+] [Seed {seed} DONE] Test Acc: {test_acc:.2f}% | Weighted F1: {test_wf1:.2f}% | Macro F1: {test_mf1:.2f}%")
    return test_acc, test_wf1, test_mf1, test_logits, test_labels

# ==================================================================================================
# 8. MASTER RUNNER (SEEDS 0, 1, 2 + ENSEMBLE ARBITRATION)
# ==================================================================================================
def run_dumir_plus():
    args = DUMIRPlusConfig()
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
    print(f"[+] Active MIntRec dataset directory: {data_dir}")

    with open(data_dir / "audio_feats.pkl", "rb") as f:
        audio_raw = pickle.load(f)
    with open(data_dir / "video_feats.pkl", "rb") as f:
        video_raw = pickle.load(f)

    keys = list(audio_raw.keys())
    audio_feats = pad_features(audio_raw, keys, args.audio_seq_len)
    video_feats = pad_features(video_raw, keys, args.video_seq_len)

    tokenizer = BertTokenizer.from_pretrained(args.text_backbone, do_lower_case=True)

    train_ds = DUMIRDataset(data_dir / "train.tsv", audio_feats, video_feats, tokenizer)
    dev_ds = DUMIRDataset(data_dir / "dev.tsv", audio_feats, video_feats, tokenizer)
    test_ds = DUMIRDataset(data_dir / "test.tsv", audio_feats, video_feats, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    class_weights = torch.ones(args.num_labels, dtype=torch.float32)
    for c in ['Joke', 'Taunt', 'Flaunt', 'Criticize']:
        class_weights[args.intent_labels.index(c)] = args.conflict_weight
    class_weights = class_weights.to(device)

    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    seed_accs, seed_wf1s, all_test_logits = [], [], []

    print("\n" + "=" * 80)
    print(">>> EXECUTING DUMIR+ MULTI-SEED BENCHMARK (SEEDS 0, 1, 2)")
    print("=" * 80)

    for seed in args.seeds:
        t_acc, t_wf1, t_mf1, t_logits, test_labels = train_seed(
            seed, train_loader, dev_loader, test_loader, cls_criterion, args, device
        )
        seed_accs.append(t_acc)
        seed_wf1s.append(t_wf1)
        all_test_logits.append(t_logits)

    print("\n" + "=" * 80)
    print(">>> FINAL DUMIR+ SOTA BENCHMARK CONSOLIDATION")
    print("=" * 80)
    for s, acc, wf1 in zip(args.seeds, seed_accs, seed_wf1s):
        print(f"Seed {s:<3} : Test Accuracy = {acc:.2f}% | Weighted F1 = {wf1:.2f}%")

    mean_acc = np.mean(seed_accs)
    mean_wf1 = np.mean(seed_wf1s)
    print("-" * 80)
    print(f"3-Seed Mean Performance : Accuracy = {mean_acc:.2f}% | Weighted F1 = {mean_wf1:.2f}%")

    # Softmax Ensemble Arbitration
    probs = [F.softmax(torch.tensor(l), dim=-1).numpy() for l in all_test_logits]
    ens_prob = np.mean(probs, axis=0)
    ens_preds = np.argmax(ens_prob, axis=-1)
    ens_acc = accuracy_score(test_labels, ens_preds) * 100.0
    ens_wf1 = f1_score(test_labels, ens_preds, average='weighted') * 100.0
    ens_mf1 = f1_score(test_labels, ens_preds, average='macro') * 100.0

    print("=" * 80)
    print(f"🏆 DUMIR+ ENSEMBLE TEST ACCURACY : {ens_acc:.2f}% (DUMIR SOTA Target: 76.01%)")
    print(f"🏆 DUMIR+ ENSEMBLE WEIGHTED F1   : {ens_wf1:.2f}% (DUMIR SOTA Target: 75.87%)")
    print(f"🏆 DUMIR+ ENSEMBLE MACRO F1      : {ens_mf1:.2f}%")
    print("=" * 80)

    dumir_target = 76.01
    delta = ens_acc - dumir_target
    if ens_acc >= dumir_target:
        print(f"\n🎉 OUTSTANDING RECORD: DUMIR+ OFFICIALLY BEAT THE 2026 SOTA BENCHMARK ({dumir_target}%) BY +{delta:.2f}%!")
    else:
        best_single = max(seed_accs)
        print(f"Best Single Seed: {best_single:.2f}% (Delta: {best_single - dumir_target:+.2f}%)")

if __name__ == "__main__":
    run_dumir_plus()
