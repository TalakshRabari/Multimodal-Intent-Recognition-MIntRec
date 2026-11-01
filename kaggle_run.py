"""
====================================================================================================
      DUMIR-HCMI: DYNAMIC UNCERTAINTY-AWARE MULTIMODAL INTENT RECOGNITION (SOTA 2026)
====================================================================================================
Project:       Multimodal Intent Recognition (MIR)
Authors:       Talaksh Rabari & Swayam Patel (Dhirubhai Ambani University)
Architecture:  Complete DUMIR Implementation (Pattern Recognition 2026 SOTA)
               - UAGM: Uncertainty-Aware Gating Module with Statistical Penalty (alpha = 5)
               - HCMI: Hierarchical Cascaded Cross-Modal Interaction across Text, Audio, and Video
               - CMCC: Cross-Modal InfoNCE Consistency Constraint (temperature = 0.05)
               - MAS: Modality-Level Auxiliary Supervision (lambda_m = 0.80)
               - Gated Prototype Memory Fusion
Target:        Outperform the 76.01% ACC / 75.87% WF1 SOTA Benchmark on MIntRec
Dataset:       MIntRec (/kaggle/input/datasets/talakshrabari/mintrec)
Hardware:      Kaggle GPU T4 x1 (16 GB VRAM)

INSTRUCTIONS:
1. Open a NEW Kaggle Notebook.
2. Under Settings: Accelerator -> GPU T4, Internet -> ON.
3. Paste this ENTIRE file into ONE code cell and click Run (or "Save Version" -> "Run all").
====================================================================================================
"""

import __main__
if not hasattr(__main__, '__file__'):
    __main__.__file__ = '/kaggle/working/dumir_run.py'

import os
import sys
import math
import copy
import csv
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import transformers
transformers.logging.set_verbosity_error()

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==================================================================================================
# 1. CONFIGURATION (Matching DUMIR Table 2 & Section 3.4.2)
# ==================================================================================================
class Config:
    dataset = "MIntRec"
    text_backbone = "bert-base-uncased"
    
    # Feature dimensions & sequence lengths
    text_feat_dim = 768
    audio_feat_dim = 768
    video_feat_dim = 1024
    text_seq_len = 30
    audio_seq_len = 480
    video_seq_len = 230
    
    # DUMIR Architectural Parameters (Table 2)
    hidden_dim = 360                  # DUMIR hidden dimension = 360
    nheads = 4                        # Attention heads = 4
    n_levels = 4                      # Cross-attention layers = 4
    alpha = 5.0                       # Uncertainty penalty factor (Fig. 3)
    temperature = 0.05                # InfoNCE temperature = 0.05
    top_k_prototypes = 3
    
    # DUMIR Multi-Task Loss Weights (Section 3.4.2)
    lambda_m = 0.80                   # Modality auxiliary supervision weight
    lambda_c = 0.03                   # Consistency contrastive loss weight
    
    # DUMIR Regularization Hyperparameters (Table 2)
    text_dropout = 0.4                # Text dropout = 0.4
    output_dropout = 0.2              # Output dropout = 0.2
    embed_dropout = 0.1               # Embedding dropout = 0.1
    attn_dropout = 0.1
    grad_clip = 0.5                   # Gradient clipping = 0.5
    
    # Training Configuration (Overnight 5-Seed Run)
    seeds = [0, 1, 2, 3, 4]           # 5-seed evaluation for thesis
    num_train_epochs = 20             # 20 Epochs with early stopping
    wait_patience = 8                 # Early stopping patience = 8 (Table 2)
    train_batch_size = 16
    eval_batch_size = 16
    test_batch_size = 16
    lr = 3e-5                         # DUMIR learning rate = 3e-5 (Table 2)
    weight_decay = 0.01
    warmup_proportion = 0.1
    
    # 20 fine-grained intent classes
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help' 
    ]
    num_labels = len(intent_labels)

args = Config()

# ==================================================================================================
# 2. HARDWARE & RUNTIME ENVIRONMENT
# ==================================================================================================
print("=" * 80)
print(">>> [1/5] HARDWARE & RUNTIME ENVIRONMENT")
print("=" * 80)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Compute Device   : {device}")
if device.type == "cuda":
    print(f"[*] GPU Model        : {torch.cuda.get_device_name(0)}")
    print(f"[*] GPU Memory Total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"[*] PyTorch Version  : {torch.__version__}")

# ==================================================================================================
# 3. DATASET LOADING & TENSOR PIPELINE
# ==================================================================================================
print("\n" + "=" * 80)
print(">>> [2/5] LOCATING & LOADING MINTREC DATASET")
print("=" * 80)

candidate_paths = [
    Path("/kaggle/input/datasets/talakshrabari/mintrec/mintrec-data/MIntRec"),
    Path("/kaggle/input/mintrec/mintrec-data/MIntRec"),
    Path("./datasets/MIntRec")
]

mintrec_dir = None
for p in candidate_paths:
    if p.exists() and (p / "train.tsv").exists():
        mintrec_dir = p
        break

if mintrec_dir is None:
    raise FileNotFoundError("MIntRec dataset not found! Please attach the MIntRec dataset to Kaggle.")

print(f"[+] Verified MIntRec dataset directory at: {mintrec_dir}")

def build_dataset(mintrec_dir, args):
    tokenizer = BertTokenizer.from_pretrained(args.text_backbone)
    label_map = {name: idx for idx, name in enumerate(args.intent_labels)}
    splits = ["train", "dev", "test"]
    split_rows, split_indices, split_label_ids = {}, {}, {}

    for s in splits:
        tsv_path = mintrec_dir / f"{s}.tsv"
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)
            split_rows[s] = rows
            split_indices[s] = ['_'.join([r[0], r[1], r[2]]) for r in rows]
            split_label_ids[s] = [label_map[r[4]] for r in rows]
        print(f"    - Loaded {s:<5}: {len(rows)} samples")

    print(f"[*] Loading video pickle: video_feats.pkl...")
    with open(mintrec_dir / "video_feats.pkl", "rb") as f:
        video_data = pickle.load(f)

    print(f"[*] Loading audio pickle: audio_feats.pkl...")
    with open(mintrec_dir / "audio_feats.pkl", "rb") as f:
        audio_data = pickle.load(f)

    def pad_features(raw_feats, keys, max_len):
        feats = []
        for k in keys:
            raw = raw_feats[k]
            arr = raw.cpu().numpy() if hasattr(raw, 'cpu') else np.array(raw, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr.squeeze(1)
            L = arr.shape[0]
            if L > max_len:
                arr = arr[:max_len]
            else:
                arr = np.pad(arr, ((0, max_len - L), (0, 0)))
            feats.append(arr)
        return np.array(feats, dtype=np.float32)

    text_data = {}
    for s in splits:
        rows = split_rows[s]
        input_ids_list, mask_list = [], []
        for row in rows:
            utterance = row[3]
            enc = tokenizer(
                utterance,
                max_length=args.text_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(enc['input_ids'].squeeze(0))
            mask_list.append(enc['attention_mask'].squeeze(0))
        text_data[s] = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(mask_list)
        }

    video_padded = {s: pad_features(video_data, split_indices[s], args.video_seq_len) for s in splits}
    audio_padded = {s: pad_features(audio_data, split_indices[s], args.audio_seq_len) for s in splits}

    class MIntRecDataset(Dataset):
        def __init__(self, label_ids, text_dict, video_f, audio_f):
            self.label_ids = label_ids
            self.input_ids = text_dict['input_ids']
            self.attention_mask = text_dict['attention_mask']
            self.video_f = video_f
            self.audio_f = audio_f

        def __len__(self):
            return len(self.label_ids)

        def __getitem__(self, i):
            return {
                'label_ids': torch.tensor(self.label_ids[i], dtype=torch.long),
                'input_ids': self.input_ids[i],
                'attention_mask': self.attention_mask[i],
                'video_feats': torch.tensor(self.video_f[i], dtype=torch.float32),
                'audio_feats': torch.tensor(self.audio_f[i], dtype=torch.float32),
            }

    return {
        s: MIntRecDataset(split_label_ids[s], text_data[s], video_padded[s], audio_padded[s])
        for s in splits
    }

datasets = build_dataset(mintrec_dir, args)

# ==================================================================================================
# 4. NEURAL MODULES: UAGM, HCMI & PROTOTYPE MEMORY
# ==================================================================================================
print("\n" + "=" * 80)
print(">>> [3/5] BUILDING DUMIR-HCMI NEURAL ARCHITECTURE")
print("=" * 80)

class UncertaintyAwareGating(nn.Module):
    """
    Uncertainty-Aware Gating Module (UAGM) from DUMIR (Section 3.2, Eqs. 6-10).
    Estimates statistical reliability from sequence variance and applies an uncertainty penalty.
    """
    def __init__(self, hidden_dim, alpha=5.0):
        super().__init__()
        self.alpha = alpha
        self.saliency_proj = nn.Linear(hidden_dim * 3, 3)

    def forward(self, H_l, H_a, H_v):
        # 1. First-order statistics (centroids / means)
        g_l = torch.mean(H_l, dim=1)
        g_a = torch.mean(H_a, dim=1)
        g_v = torch.mean(H_v, dim=1)

        # 2. Second-order statistics (variances across time)
        v_l = torch.mean((H_l - g_l.unsqueeze(1)) ** 2, dim=[1, 2])
        v_a = torch.mean((H_a - g_a.unsqueeze(1)) ** 2, dim=[1, 2])
        v_v = torch.mean((H_v - g_v.unsqueeze(1)) ** 2, dim=[1, 2])

        u_l = torch.sigmoid(v_l).unsqueeze(-1)
        u_a = torch.sigmoid(v_a).unsqueeze(-1)
        u_v = torch.sigmoid(v_v).unsqueeze(-1)
        u = torch.cat([u_l, u_a, u_v], dim=-1)  # (B, 3)

        # 3. Saliency score with uncertainty penalty: S_m = e_m - alpha * u_m (Eq. 8)
        g_joint = torch.cat([g_l, g_a, g_v], dim=-1)
        e = self.saliency_proj(g_joint)
        S = e - self.alpha * u
        weights = F.softmax(S, dim=-1)  # (B, 3)

        w_l = weights[:, 0].view(-1, 1, 1)
        w_a = weights[:, 1].view(-1, 1, 1)
        w_v = weights[:, 2].view(-1, 1, 1)

        F_l = w_l * H_l
        F_a = w_a * H_a
        F_v = w_v * H_v

        return F_l, F_a, F_v, weights

class CrossModalAttentionBlock(nn.Module):
    """Cross-Modal Attention Operator: CA(Q, K, V) from DUMIR (Eq. 11)."""
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
        x = self.norm2(x + self.ffn(x))
        return x

class HCMI(nn.Module):
    """
    Hierarchical Cross-Modal Interaction (HCMI) from DUMIR (Section 3.3, Eqs. 12-14).
    Progressively models cascaded cross-modal interactions for Text, Audio, and Video anchors.
    """
    def __init__(self, hidden_dim=360, nheads=4, dropout=0.1):
        super().__init__()
        # Text Anchor Branch
        self.ca_la = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_lav = CrossModalAttentionBlock(hidden_dim, nheads, dropout)

        # Audio Anchor Branch
        self.ca_al = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_alv = CrossModalAttentionBlock(hidden_dim, nheads, dropout)

        # Video Anchor Branch
        self.ca_vl = CrossModalAttentionBlock(hidden_dim, nheads, dropout)
        self.ca_vla = CrossModalAttentionBlock(hidden_dim, nheads, dropout)

    def forward(self, F_l, F_a, F_v):
        # 1. Text Anchor: H_l<-a -> H_l<-(a,v) -> Z_l (Eqs. 12-14)
        H_la = self.ca_la(F_l, F_a, F_a)
        H_lav = self.ca_lav(H_la, F_v, F_v)
        Z_l = torch.cat([torch.mean(H_la, dim=1), torch.mean(H_lav, dim=1)], dim=-1)

        # 2. Audio Anchor: H_a<-l -> H_a<-(l,v) -> Z_a
        H_al = self.ca_al(F_a, F_l, F_l)
        H_alv = self.ca_alv(H_al, F_v, F_v)
        Z_a = torch.cat([torch.mean(H_al, dim=1), torch.mean(H_alv, dim=1)], dim=-1)

        # 3. Video Anchor: H_v<-l -> H_v<-(l,a) -> Z_v
        H_vl = self.ca_vl(F_v, F_l, F_l)
        H_vla = self.ca_vla(H_vl, F_a, F_a)
        Z_v = torch.cat([torch.mean(H_vl, dim=1), torch.mean(H_vla, dim=1)], dim=-1)

        return Z_l, Z_a, Z_v

class PrototypeMemory(nn.Module):
    """Class Prototype Memory to align fine-grained intent boundaries."""
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

class Conv1dBlock(nn.Module):
    """Conv1D temporal feature projection with LayerNorm and Dropout (Table 2)."""
    def __init__(self, in_dim, out_dim, kernel_size=1, padding=0, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, L, in_dim) -> transpose to (B, in_dim, L) for Conv1d -> transpose back
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.norm(out))

class DUMIR_HCMI(nn.Module):
    """
    Unified DUMIR-HCMI Model.
    Integrates Pretrained BERT, Conv1D Feature Projections, UAGM Uncertainty Gating,
    HCMI Cascaded Cross-Modal Attention, Modality Auxiliary Heads, and Prototype Memory.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Backbones & Input Projections
        self.bert = BertModel.from_pretrained(args.text_backbone)
        self.text_dropout = nn.Dropout(args.text_dropout)

        # Conv1D feature projections (Table 2: 5/1/1 kernels)
        self.proj_l = Conv1dBlock(args.text_feat_dim, args.hidden_dim, kernel_size=5, padding=2, dropout=args.embed_dropout)
        self.proj_a = Conv1dBlock(args.audio_feat_dim, args.hidden_dim, kernel_size=1, dropout=args.embed_dropout)
        self.proj_v = Conv1dBlock(args.video_feat_dim, args.hidden_dim, kernel_size=1, dropout=args.embed_dropout)

        # Core Modules
        self.uagm = UncertaintyAwareGating(args.hidden_dim, alpha=args.alpha)
        self.hcmi = HCMI(args.hidden_dim, nheads=args.nheads, dropout=args.attn_dropout)

        # Joint Multimodal Fusion & Classification (Eq. 15)
        joint_dim = args.hidden_dim * 2 * 3  # 360 * 2 * 3 = 2160
        self.fusion_mlp = nn.Sequential(
            nn.Linear(joint_dim, 768),
            nn.GELU(),
            nn.Dropout(args.output_dropout)
        )
        self.proto_memory = PrototypeMemory(768, args.num_labels, args.top_k_prototypes)
        self.classifier = nn.Linear(768, args.num_labels)
        self.out_dropout = nn.Dropout(args.output_dropout)

        # Modality Auxiliary Heads (MAS, Eq. 20)
        anchor_dim = args.hidden_dim * 2  # 720
        self.aux_head_l = nn.Linear(anchor_dim, args.num_labels)
        self.aux_head_a = nn.Linear(anchor_dim, args.num_labels)
        self.aux_head_v = nn.Linear(anchor_dim, args.num_labels)

    def forward(self, input_ids, attention_mask, audio_feats, video_feats):
        # 1. Text Backbone Representation
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        bert_out = self.text_dropout(bert_out)

        # 2. Conv1D Projections to uniform hidden_dim (360)
        H_l = self.proj_l(bert_out)
        H_a = self.proj_a(audio_feats)
        H_v = self.proj_v(video_feats)

        # 3. Uncertainty-Aware Gating (UAGM)
        F_l, F_a, F_v, u_weights = self.uagm(H_l, H_a, H_v)

        # 4. Hierarchical Cross-Modal Interaction (HCMI)
        Z_l, Z_a, Z_v = self.hcmi(F_l, F_a, F_v)

        # 5. Joint Fusion & Prototype Refinement
        Z_joint = torch.cat([Z_l, Z_a, Z_v], dim=-1)  # (B, 2160)
        fused = self.fusion_mlp(Z_joint)
        enhanced = self.proto_memory(fused)
        logits = self.classifier(self.out_dropout(enhanced))

        # Unimodal Auxiliary Predictions
        aux_logits = {
            'text': self.aux_head_l(Z_l),
            'audio': self.aux_head_a(Z_a),
            'video': self.aux_head_v(Z_v)
        }

        return logits, Z_l, Z_a, Z_v, aux_logits

# ==================================================================================================
# 5. MULTI-TASK LOSS: FUSION + MAS + CMCC InfoNCE (Eq. 16)
# ==================================================================================================
def compute_infonce(z1, z2, temperature=0.05):
    """Pairwise InfoNCE Consistency Loss from DUMIR (Eq. 18)."""
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    sim = torch.matmul(z1_norm, z2_norm.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0
    return loss

# ==================================================================================================
# 6. EVALUATION & TRAINING ENGINE
# ==================================================================================================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            video_f = batch['video_feats'].to(device)
            audio_f = batch['audio_feats'].to(device)
            labels = batch['label_ids']

            logits, _, _, _, _ = model(input_ids, attn_mask, audio_f, video_f)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    wf1 = f1_score(all_labels, all_preds, average='weighted') * 100
    mf1 = f1_score(all_labels, all_preds, average='macro') * 100
    return acc, wf1, mf1

def train_single_seed(seed, datasets, args, device):
    print(f"\n>>> Running Seed {seed} (Up to {args.num_train_epochs} Epochs with Patience {args.wait_patience})...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(datasets['train'], batch_size=args.train_batch_size, shuffle=True, drop_last=False)
    dev_loader = DataLoader(datasets['dev'], batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(datasets['test'], batch_size=args.test_batch_size, shuffle=False)

    model = DUMIR_HCMI(args).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    total_steps = len(train_loader) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    cls_criterion = nn.CrossEntropyLoss()

    best_dev_score = 0.0
    best_model_weights = None
    best_epoch = 0
    patience = 0

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            video_f = batch['video_feats'].to(device)
            audio_f = batch['audio_feats'].to(device)
            labels = batch['label_ids'].to(device)

            logits, Z_l, Z_a, Z_v, aux_logits = model(input_ids, attn_mask, audio_f, video_f)

            # 1. Main Fusion Loss (Eq. 17)
            loss_fusion = cls_criterion(logits, labels)

            # 2. Modality-Level Auxiliary Supervision (MAS, Eq. 20)
            loss_mod = (
                cls_criterion(aux_logits['text'], labels) +
                cls_criterion(aux_logits['audio'], labels) +
                cls_criterion(aux_logits['video'], labels)
            ) / 3.0

            # 3. Cross-Modal Consistency InfoNCE Loss (CMCC, Eq. 18-19)
            loss_cons = (
                compute_infonce(Z_l, Z_a, args.temperature) +
                compute_infonce(Z_l, Z_v, args.temperature) +
                compute_infonce(Z_a, Z_v, args.temperature)
            ) / 3.0

            # Total Multi-Task Loss: L = L_fusion + 0.8 * L_mod + 0.03 * L_cons (Eq. 16)
            loss = loss_fusion + args.lambda_m * loss_mod + args.lambda_c * loss_cons

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dev_acc, dev_wf1, dev_mf1 = evaluate(model, dev_loader, device)

        print(f"  [Seed {seed}] Ep {epoch:2d}/{args.num_train_epochs:2d} | Loss: {avg_loss:.3f} | Dev Acc: {dev_acc:.2f}% | Dev WF1: {dev_wf1:.2f}% | Dev MF1: {dev_mf1:.2f}%")

        # Monitor Dev WF1 matching DUMIR Sec. 4.2
        if dev_wf1 > best_dev_score:
            best_dev_score = dev_wf1
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= args.wait_patience:
                print(f"  [*] Early stopping at Epoch {epoch} (Peak Dev WF1: {best_dev_score:.2f}% at Ep {best_epoch})")
                break

    model.load_state_dict(best_model_weights)
    test_acc, test_wf1, test_mf1 = evaluate(model, test_loader, device)

    # Save best checkpoint
    save_path = f"/kaggle/working/dumir_hcmi_seed_{seed}.pt"
    try:
        torch.save(best_model_weights, save_path)
        print(f"  [*] Saved checkpoint to {save_path}")
    except Exception:
        pass

    print(f"[+] [Seed {seed} DONE] Test Acc: {test_acc:.2f}% | Weighted F1: {test_wf1:.2f}% | Macro F1: {test_mf1:.2f}%")
    return {"acc": test_acc, "wf1": test_wf1, "macro_f1": test_mf1}

# ==================================================================================================
# 7. MULTI-SEED RUN EXECUTION & RESULTS COMPILATION
# ==================================================================================================
print("\n" + "=" * 80)
print(f">>> [4/5] EXECUTING 5-SEED BENCHMARK EVALUATION: {args.seeds}")
print("=" * 80)

run_results = {}
for seed in args.seeds:
    run_results[seed] = train_single_seed(seed, datasets, args, device)

print("\n" + "=" * 80)
print(">>> [5/5] FINAL CONSOLIDATED PERFORMANCE SUMMARY")
print("=" * 80)
print(f"{'Seed':<8} | {'Accuracy (%)':<16} | {'Weighted F1 (%)':<18} | {'Macro F1 (%)':<16}")
print("-" * 70)

accs, wf1s, mf1s = [], [], []
for seed, r in run_results.items():
    accs.append(r["acc"])
    wf1s.append(r["wf1"])
    mf1s.append(r["macro_f1"])
    print(f"Seed {seed:<3} | {r['acc']:>12.2f} %     | {r['wf1']:>14.2f} %     | {r['macro_f1']:>12.2f} %")

print("-" * 70)
mean_acc = np.mean(accs)
mean_wf1 = np.mean(wf1s)
mean_mf1 = np.mean(mf1s)
print(f"{'Mean':<8} | {mean_acc:>12.2f} %     | {mean_wf1:>14.2f} %     | {mean_mf1:>12.2f} %")
print("=" * 80)

try:
    with open('/kaggle/working/dumir_hcmi_results.txt', 'w') as f:
        f.write(f"DUMIR-HCMI Final Benchmark Results across seeds {args.seeds}\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}%\nMean Weighted F1: {mean_wf1:.2f}%\nMean Macro F1: {mean_mf1:.2f}%\n")
    print("  [+] Saved final benchmark report to /kaggle/working/dumir_hcmi_results.txt")
except Exception:
    pass

reprod_acc = 73.48
dumir_sota = 76.01
delta_base = mean_acc - reprod_acc
delta_sota = mean_acc - dumir_sota

print(f"[*] Baseline Reproduced ACC      : {reprod_acc:.2f}% (Delta: {delta_base:+.2f}%)")
print(f"[*] 2026 DUMIR SOTA Benchmark ACC: {dumir_sota:.2f}% (Delta: {delta_sota:+.2f}%)")
print(f"[*] Current DUMIR-HCMI Mean ACC  : {mean_acc:.2f}%")

if mean_acc >= dumir_sota:
    print("\n>>> 🏆 OUTSTANDING SUCCESS: DUMIR-HCMI OUTPERFORMED THE 2026 SOTA BENCHMARK (76.01%)!")
elif mean_acc > reprod_acc:
    print(f"\n>>> EXCELLENT PROGRESS: DUMIR-HCMI beat the baseline by +{delta_base:.2f}%!")
print("=" * 80)
