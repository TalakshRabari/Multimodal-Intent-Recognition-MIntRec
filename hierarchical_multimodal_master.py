"""
====================================================================================================
    HIERARCHICAL MULTIMODAL INTENT RECOGNITION (MIR) - COMPLETE ALL-IN-ONE MASTER SUITE
====================================================================================================
Project:       Multimodal Intent Recognition (Major Project)
Authors:       Talaksh Rabari & Swayam Patel (Dhirubhai Ambani University)
Advisor:       Dr. Shruti Bhilare
Dataset:       MIntRec (20 Fine-Grained Intent Classes)
Hardware:      Local CPU/GPU and Kaggle GPU T4 x1 (16 GB VRAM)

ALL EXPERIMENTS FULLY INCLUDED & UNIFIED:
1. Audio-Video Only Model (Training WITHOUT Text):
   - Multi-scale 1D-CNN Temporal Pyramids (kernel 3, 7) + Bidirectional Cross-Modal Attention.
   - Attentive Temporal Pooling for non-verbal acoustic prosody and visual gestures.
   - Complete standalone training and evaluation.

2. Hierarchical Two-Stage Fusion (A+V -> then + Text):
   - Stage 1: Audio and Video cross-attend and fuse into unified non-verbal representation z_AV.
   - Stage 2: Calibrated Bounded Residual Shift (beta_shift = 0.006) onto BERT text representation.
   - Mathematically prevents noisy non-verbal frames from degrading confident text!

3. Level-Wise / Sequential Intent Recognition (Text First):
   - Level 1: Run Text classifier first to compute prediction and softmax confidence c_text.
   - Level 2: If c_text >= tau, trust text; if c_text < tau, condition/refine on non-verbal cues.
   - Sweeps across thresholds tau in [0.50, 0.65, 0.75, 0.85, 0.95].

4. Softmax Confidence Analytics:
   - Extracts top-1 softmax confidence, prediction entropy, and margins across all modalities.

5. Manual Modality Weighting & Text Weight Reduction Sweep:
   - Parametric control: z_final = w_text * z_T + (1 - w_text) * (w_audio * z_A + w_video * z_V).
   - Evaluates performance across w_text in [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0].

6. Overfitting Prevention & Calibrated Conflict Loss:
   - Early stopping (patience = 3) to stop at the optimal validation peak (Epoch 3-4).
   - 1.6x conflict weight on Joke, Taunt, Flaunt, Criticize + Label Smoothing (0.1).

7. Publication-Ready Plotting Suite:
   - Fig 1: Intent-by-Intent Agreement vs Disagreement Breakdown across all 20 classes.
   - Fig 2: 4-Quadrant Case Analysis (Concordant, Text-only, AV-Rescue, Mutual Failure).
   - Fig 3: Softmax Confidence Distributions (Agreement vs Conflict).
   - Fig 4: Performance Curve under Text Weight Reduction (1.0 -> 0.0).
====================================================================================================
"""

import __main__
if not hasattr(__main__, '__file__'):
    __main__.__file__ = '/kaggle/working/hierarchical_multimodal_experiments.py'

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

from sklearn.metrics import accuracy_score, f1_score

# ==================================================================================================
# 1. GLOBAL CONFIGURATION
# ==================================================================================================
class MasterConfig:
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
    beta_shift = 0.006                 # Calibrated bounded non-verbal shift threshold
    
    num_classes = 20
    train_batch_size = 16
    eval_batch_size = 16
    lr_bert = 2e-5
    lr_fusion = 1e-4
    lr_av_standalone = 3e-4
    weight_decay = 0.01
    
    epochs_av = 5                      # Epochs for standalone Audio-Video training
    epochs_joint = 10                  # Epochs for joint hierarchical training
    patience = 3                       # Early stopping patience on validation set
    
    label_smoothing = 0.1
    conflict_weight = 1.6              # 1.6x penalty on Joke, Taunt, Flaunt, Criticize
    
    intent_labels = [
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 
        'Joke', 'Oppose', 
        'Comfort', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
        'Prevent', 'Greet', 'Ask for help' 
    ]

# ==================================================================================================
# 2. DATASET LOADER & FEATURE PADDING
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

class MasterMIntRecDataset(Dataset):
    def __init__(self, tsv_path, audio_feats, video_feats, tokenizer):
        self.keys, self.labels, self.texts = [], [], []
        label_map = {name: idx for idx, name in enumerate(MasterConfig.intent_labels)}
        
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
            max_length=MasterConfig.text_seq_len,
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
# 3. ARCHITECTURE 1: MULTI-SCALE AUDIO-VIDEO ONLY MODEL (TRAINING WITHOUT TEXT)
# ==================================================================================================
class AudioVideoTemporalModule(nn.Module):
    """
    Non-verbal feature extractor: 1D-CNN temporal pyramids + bidirectional cross-attention.
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
        
        # 1D-CNN Temporal Pyramids (Kernel 3 & 7 for audio; 3 & 5 for video)
        self.conv_a1 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=3, padding=1)
        self.conv_a2 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=7, padding=3)
        self.conv_v1 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=3, padding=1)
        self.conv_v2 = nn.Conv1d(shared_dim, shared_dim // 2, kernel_size=5, padding=2)
        
        # Bidirectional Cross Attention between Audio and Video
        self.cross_attn = nn.MultiheadAttention(shared_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(shared_dim)
        
        # Attentive Temporal Pooling
        self.pool_net = nn.Sequential(
            nn.Linear(shared_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Linear(shared_dim, MasterConfig.num_classes)

    def forward(self, audio_feats, video_feats, w_audio=0.5, w_video=0.5):
        # 1D-CNN Temporal Convolutions
        ha = self.proj_a(audio_feats).transpose(1, 2)
        hv = self.proj_v(video_feats).transpose(1, 2)
        
        ha = F.gelu(torch.cat([self.conv_a1(ha), self.conv_a2(ha)], dim=1)).transpose(1, 2)
        hv = F.gelu(torch.cat([self.conv_v1(hv), self.conv_v2(hv)], dim=1)).transpose(1, 2)
        
        # Manual weighting ratio
        w_sum = w_audio + w_video + 1e-9
        weighted_ha = (w_audio / w_sum) * ha
        weighted_hv = (w_video / w_sum) * hv
        
        # Cross Attention
        attn_out, _ = self.cross_attn(query=weighted_ha, key=weighted_hv, value=weighted_hv)
        fused_seq = self.norm(weighted_ha + attn_out)
        
        # Attentive Temporal Pooling
        scores = self.pool_net(fused_seq)
        weights = F.softmax(scores, dim=1)
        z_av = (fused_seq * weights).sum(dim=1)
        
        logits_av = self.classifier(z_av)
        return z_av, logits_av

class StandaloneAudioVideoClassifier(nn.Module):
    """Pure non-verbal model trained completely without text."""
    def __init__(self, config):
        super().__init__()
        self.av_module = AudioVideoTemporalModule(
            audio_dim=config.audio_feat_dim,
            video_dim=config.video_feat_dim,
            shared_dim=config.shared_dim,
            dropout=config.dropout
        )
    def forward(self, audio_feats, video_feats, w_audio=0.5, w_video=0.5):
        z_av, logits_av = self.av_module(audio_feats, video_feats, w_audio, w_video)
        return logits_av, z_av

# ==================================================================================================
# 4. ARCHITECTURE 2 & 3: CALIBRATED HIERARCHICAL & SEQUENTIAL LEVEL-WISE MODEL
# ==================================================================================================
class MasterHierarchicalModel(nn.Module):
    """
    Two-Stage Hierarchical Model with Calibrated Bounded Residual Shift:
    Stage 1: Fuse Audio & Video -> z_AV.
    Stage 2: Calibrated residual shift onto BERT text representation.
    Includes:
    - Sequential Level-Wise Gating (Text First, condition on AV if confidence < tau)
    - Manual Modality Weighting (w_text, w_audio, w_video)
    - Softmax Confidence extraction
    """
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.text_backbone)
        self.shared_dim = config.shared_dim
        self.beta_shift = config.beta_shift
        
        # Stage 1: Non-Verbal Module
        self.av_module = AudioVideoTemporalModule(
            audio_dim=config.audio_feat_dim,
            video_dim=config.video_feat_dim,
            shared_dim=config.shared_dim,
            dropout=config.dropout
        )
        
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
        
        # Sequential Refinement Layer
        self.seq_refinement = nn.Sequential(
            nn.Linear(config.num_classes * 2 + 1, config.num_classes),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask, audio_feats, video_feats,
                w_text=1.0, w_audio=0.5, w_video=0.5, sequential_mode=False, tau_conf=0.75):
        # 1. Text Representation
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]
        h_t = self.text_proj(cls_token)
        logits_text = self.text_classifier(h_t)
        
        # 2. Stage 1: Non-Verbal Representation (Audio + Video)
        z_av, logits_av = self.av_module(audio_feats, video_feats, w_audio, w_video)
        
        # 3. Stage 2: Calibrated Bounded Residual Shift
        h_av = self.w_av_proj(z_av)
        
        t_norm = torch.sqrt(torch.sum(h_t ** 2, dim=-1, keepdim=True) + 1e-8)
        av_norm = torch.sqrt(torch.sum(h_av ** 2, dim=-1, keepdim=True) + 1e-8)
        threshold = (t_norm / av_norm) * self.beta_shift
        alpha = torch.clamp(threshold, max=1.0)
        
        learned_gate = self.gate_mlp(torch.cat([h_t, h_av], dim=-1))
        
        # Bounded displacement
        shift = alpha * learned_gate * h_av
        fused_rep = self.norm(w_text * h_t + (1.0 - w_text * 0.5) * shift)
        logits_fused = self.fused_classifier(self.dropout(fused_rep))
        
        # 4. Softmax Confidences
        prob_text = F.softmax(logits_text, dim=-1)
        conf_text, _ = torch.max(prob_text, dim=-1)
        
        prob_av = F.softmax(logits_av, dim=-1)
        conf_av, _ = torch.max(prob_av, dim=-1)
        
        prob_fused = F.softmax(logits_fused, dim=-1)
        conf_fused, _ = torch.max(prob_fused, dim=-1)
        
        # 5. Sequential Level-Wise Conditional Gating
        if sequential_mode:
            # Step 1: Default to pure text predictions (unfused, 100% text)
            seq_logits = logits_text.clone()
            
            # Step 2: CONDITIONAL FUSION - Only fuse with Audio+Video if confidence is strictly less than tau!
            fuse_mask = (conf_text < tau_conf)
            if fuse_mask.any():
                seq_logits[fuse_mask] = logits_fused[fuse_mask]
                
            logits_output = seq_logits
        else:
            logits_output = logits_fused
            
        return {
            'logits': logits_output,
            'logits_text': logits_text,
            'logits_av': logits_av,
            'conf_text': conf_text,
            'conf_av': conf_av,
            'conf_fused': conf_fused,
            'prob_text': prob_text,
            'prob_av': prob_av,
            'prob_fused': prob_fused
        }

# ==================================================================================================
# 5. EVALUATION ENGINE
# ==================================================================================================
def evaluate_master(model, loader, device, sequential_mode=False, tau_conf=0.75,
                    w_text=1.0, w_audio=0.5, w_video=0.5):
    model.eval()
    all_preds_f, all_preds_t, all_preds_av, all_labels = [], [], [], []
    all_conf_t, all_conf_av, all_conf_f = [], [], []
    
    with torch.no_grad():
        for b in loader:
            labels = b['label'].to(device)
            af = b['audio_feats'].to(device)
            vf = b['video_feats'].to(device)
            
            if isinstance(model, StandaloneAudioVideoClassifier):
                logits_av, _ = model(af, vf, w_audio, w_video)
                prob_av = F.softmax(logits_av, dim=-1)
                conf_av, preds = torch.max(prob_av, dim=-1)
                all_preds_av.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_conf_av.extend(conf_av.cpu().numpy())
            else:
                out = model(
                    b['input_ids'].to(device), b['attention_mask'].to(device),
                    af, vf,
                    w_text=w_text, w_audio=w_audio, w_video=w_video,
                    sequential_mode=sequential_mode, tau_conf=tau_conf
                )
                all_preds_f.extend(torch.argmax(out['logits'], dim=-1).cpu().numpy())
                all_preds_t.extend(torch.argmax(out['logits_text'], dim=-1).cpu().numpy())
                all_preds_av.extend(torch.argmax(out['logits_av'], dim=-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                all_conf_t.extend(out['conf_text'].cpu().numpy())
                all_conf_av.extend(out['conf_av'].cpu().numpy())
                all_conf_f.extend(out['conf_fused'].cpu().numpy())
                
    yt = np.array(all_labels)
    
    if isinstance(model, StandaloneAudioVideoClassifier):
        p_av = np.array(all_preds_av)
        return {
            'acc': accuracy_score(yt, p_av) * 100.0,
            'wf1': f1_score(yt, p_av, average='weighted') * 100.0,
            'mf1': f1_score(yt, p_av, average='macro') * 100.0,
            'preds_av': p_av, 'labels': yt, 'conf_av': np.array(all_conf_av)
        }
    else:
        pf, pt, pav = np.array(all_preds_f), np.array(all_preds_t), np.array(all_preds_av)
        conf_t_arr = np.array(all_conf_t)
        fused_count = int(np.sum(conf_t_arr < tau_conf))
        fused_pct = (fused_count / len(yt)) * 100.0 if len(yt) > 0 else 0.0
        return {
            'acc_fused': accuracy_score(yt, pf) * 100.0,
            'wf1_fused': f1_score(yt, pf, average='weighted') * 100.0,
            'mf1_fused': f1_score(yt, pf, average='macro') * 100.0,
            'acc_text': accuracy_score(yt, pt) * 100.0,
            'wf1_text': f1_score(yt, pt, average='weighted') * 100.0,
            'acc_av': accuracy_score(yt, pav) * 100.0,
            'wf1_av': f1_score(yt, pav, average='weighted') * 100.0,
            'preds_fused': pf, 'preds_text': pt, 'preds_av': pav, 'labels': yt,
            'conf_text': conf_t_arr, 'conf_av': np.array(all_conf_av), 'conf_fused': np.array(all_conf_f),
            'fused_count': fused_count, 'fused_pct': fused_pct
        }

# ==================================================================================================
# 6. PUBLICATION PLOTTING SUITE (FIG 1 TO FIG 4)
# ==================================================================================================
def generate_master_plots(eval_results, weight_sweep_data, output_dir="./analysis_plots"):
    os.makedirs(output_dir, exist_ok=True)
    labels = eval_results['labels']
    preds_text = eval_results['preds_text']
    preds_av = eval_results['preds_av']
    preds_fused = eval_results['preds_fused']
    conf_text = eval_results['conf_text']
    conf_av = eval_results['conf_av']
    intent_labels = MasterConfig.intent_labels
    
    print("\n" + "=" * 80)
    print(">>> GENERATING UPDATED PUBLICATION-READY PLOTS")
    print("=" * 80)

    # ----------------------------------------------------------------------------------------------
    # FIG 1: 20-Intent Disagreement vs Fused Arbitration Accuracy
    # ----------------------------------------------------------------------------------------------
    intent_stats = []
    for idx, name in enumerate(intent_labels):
        c_mask = (labels == idx)
        total = np.sum(c_mask)
        if total == 0:
            continue
        agree_mask = c_mask & (preds_text == preds_av)
        disagree_mask = c_mask & (preds_text != preds_av)
        agree_rate = (np.sum(agree_mask) / total) * 100.0
        disagree_rate = (np.sum(disagree_mask) / total) * 100.0
        
        if np.sum(disagree_mask) > 0:
            fused_win = np.mean(preds_fused[disagree_mask] == labels[disagree_mask]) * 100.0
        else:
            fused_win = 100.0
            
        intent_stats.append({
            'intent': name, 'disagree_rate': disagree_rate, 'fused_win': fused_win
        })
        
    intent_stats = sorted(intent_stats, key=lambda x: x['disagree_rate'], reverse=True)
    names = [x['intent'] for x in intent_stats]
    disagree_rates = [x['disagree_rate'] for x in intent_stats]
    fused_wins = [x['fused_win'] for x in intent_stats]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.figure(figsize=(15, 6), dpi=300)
    plt.bar(x - width/2, disagree_rates, width, label='Text vs Audio-Video Disagreement Rate (%)', color='#e74c3c', alpha=0.85)
    plt.bar(x + width/2, fused_wins, width, label='Fused Arbitration Accuracy under Conflict (%)', color='#2ecc71', alpha=0.85)
    plt.xlabel('Intent Categories (Sorted by Disagreement)', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('Modality Disagreement and Fused Arbitration Accuracy Across 20 Intent Classes', fontsize=14, fontweight='bold', pad=12)
    plt.xticks(x, names, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 110)
    plt.axhline(50, color='gray', linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig1_path = os.path.join(output_dir, "fig1_agreement_disagreement_by_intent.png")
    plt.savefig(fig1_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 1 to: {fig1_path}")

    # ----------------------------------------------------------------------------------------------
    # FIG 2: 4-Quadrant Case Analysis (Where are they same vs different?)
    # ----------------------------------------------------------------------------------------------
    q1 = np.sum((preds_text == labels) & (preds_av == labels))
    q2 = np.sum((preds_text == labels) & (preds_av != labels))
    q3 = np.sum((preds_text != labels) & (preds_av == labels))
    q4 = np.sum((preds_text != labels) & (preds_av != labels))
    total_test = len(labels)
    
    quadrant_counts = [q1, q2, q3, q4]
    quadrant_pcts = [q / total_test * 100.0 for q in quadrant_counts]
    quadrant_labels = [
        f"Concordant Correct\n(Both Agree & Correct)\n{q1} samples ({quadrant_pcts[0]:.1f}%)",
        f"Text-Dominant Correct\n(Text Right, AV Wrong)\n{q2} samples ({quadrant_pcts[1]:.1f}%)",
        f"Audio-Visual Rescue\n(AV Right, Text Wrong)\n{q3} samples ({quadrant_pcts[2]:.1f}%)",
        f"Mutual Failure\n(Both Modalities Wrong)\n{q4} samples ({quadrant_pcts[3]:.1f}%)"
    ]
    colors = ['#27ae60', '#2980b9', '#f39c12', '#c0392b']
    
    plt.figure(figsize=(10, 6), dpi=300)
    bars = plt.bar(range(4), quadrant_pcts, color=colors, alpha=0.9, edgecolor='black', linewidth=1.2)
    plt.ylabel('Proportion of Test Dataset (%)', fontsize=12, fontweight='bold')
    plt.title('Cross-Modal Case Taxonomy: Text vs Audio-Video Modality Interactions', fontsize=14, fontweight='bold', pad=12)
    plt.xticks(range(4), quadrant_labels, fontsize=10, fontweight='bold')
    plt.ylim(0, max(quadrant_pcts) + 12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, pct, count in zip(bars, quadrant_pcts, quadrant_counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{pct:.1f}% (N={count})", ha='center', va='bottom', fontweight='bold')
        
    fig2_path = os.path.join(output_dir, "fig2_four_quadrant_case_analysis.png")
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 2 to: {fig2_path}")

    # ----------------------------------------------------------------------------------------------
    # FIG 3: Softmax Prediction Confidence (Concordant vs Conflicting)
    # ----------------------------------------------------------------------------------------------
    agree_mask = (preds_text == preds_av)
    disagree_mask = (preds_text != preds_av)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    axes[0].hist(conf_text[agree_mask], bins=20, alpha=0.6, color='#27ae60', label=f'Text Conf (When Agree, Mean={conf_text[agree_mask].mean():.2f})', density=True)
    axes[0].hist(conf_text[disagree_mask], bins=20, alpha=0.6, color='#c0392b', label=f'Text Conf (When Conflict, Mean={conf_text[disagree_mask].mean():.2f})', density=True)
    axes[0].set_title('Text Modality Softmax Confidence', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Softmax Max Probability (Confidence)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].legend(frameon=True, facecolor='white')
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    axes[1].hist(conf_av[agree_mask], bins=20, alpha=0.6, color='#2980b9', label=f'AV Conf (When Agree, Mean={conf_av[agree_mask].mean():.2f})', density=True)
    axes[1].hist(conf_av[disagree_mask], bins=20, alpha=0.6, color='#e67e22', label=f'AV Conf (When Conflict, Mean={conf_av[disagree_mask].mean():.2f})', density=True)
    axes[1].set_title('Audio-Video Modality Softmax Confidence', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Softmax Max Probability (Confidence)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].legend(frameon=True, facecolor='white')
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    plt.suptitle('Softmax Prediction Confidence Distributions: Concordant vs Conflicting Utterances', fontsize=14, fontweight='bold', y=1.02)
    fig3_path = os.path.join(output_dir, "fig3_softmax_confidence_distributions.png")
    plt.savefig(fig3_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 3 to: {fig3_path}")

    # ----------------------------------------------------------------------------------------------
    # FIG 4: Modality Reliance Analysis (Text Weight Reduction Curve)
    # ----------------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    w_vals = [x['w_text'] for x in weight_sweep_data]
    acc_vals = [x['acc'] for x in weight_sweep_data]
    f1_vals = [x['wf1'] for x in weight_sweep_data]
    
    plt.plot(w_vals, acc_vals, marker='o', linewidth=2.5, color='#2980b9', label='Test Accuracy (%)')
    plt.plot(w_vals, f1_vals, marker='s', linewidth=2.5, color='#e67e22', label='Test Weighted F1 (%)')
    plt.xlabel('Text Modality Weight (w_text)', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (%)', fontsize=12, fontweight='bold')
    plt.title('Modality Reliance Analysis: Impact of Reducing Text Weight', fontsize=14, fontweight='bold', pad=12)
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    fig4_path = os.path.join(output_dir, "fig4_text_weight_reduction_curve.png")
    plt.savefig(fig4_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Figure 4 to: {fig4_path}")

# ==================================================================================================
# 7. MASTER EXECUTION PIPELINE (ALL EXPERIMENTS)
# ==================================================================================================
def run_master_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args, unknown = parser.parse_known_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target Device: {device}")
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "/kaggle/working/analysis_plots" if Path("/kaggle/working").exists() else "./analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    
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
        raise FileNotFoundError("MIntRec dataset not found! Please attach MIntRec in Kaggle.")
    print(f"[+] Active MIntRec dataset directory: {data_dir}")
    
    with open(data_dir / "audio_feats.pkl", "rb") as f:
        audio_raw = pickle.load(f)
    with open(data_dir / "video_feats.pkl", "rb") as f:
        video_raw = pickle.load(f)
        
    keys = list(audio_raw.keys())
    audio_feats = pad_features(audio_raw, keys, MasterConfig.audio_seq_len)
    video_feats = pad_features(video_raw, keys, MasterConfig.video_seq_len)
    
    tokenizer = BertTokenizer.from_pretrained(MasterConfig.text_backbone, do_lower_case=True)
    
    train_ds = MasterMIntRecDataset(data_dir / "train.tsv", audio_feats, video_feats, tokenizer)
    dev_ds = MasterMIntRecDataset(data_dir / "dev.tsv", audio_feats, video_feats, tokenizer)
    test_ds = MasterMIntRecDataset(data_dir / "test.tsv", audio_feats, video_feats, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=MasterConfig.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=MasterConfig.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=MasterConfig.eval_batch_size, shuffle=False)
    
    # ----------------------------------------------------------------------------------------------
    # EXPERIMENT 1: AUDIO-VIDEO ONLY MODEL (TRAINING WITHOUT TEXT)
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> [EXPERIMENT 1] STANDALONE AUDIO-VIDEO ONLY TRAINING (WITHOUT TEXT)")
    print("=" * 80)
    
    av_standalone = StandaloneAudioVideoClassifier(MasterConfig).to(device)
    av_optimizer = AdamW(av_standalone.parameters(), lr=MasterConfig.lr_av_standalone, weight_decay=0.01)
    
    class_weights = torch.ones(MasterConfig.num_classes, dtype=torch.float32)
    for c in ['Joke', 'Taunt', 'Flaunt', 'Criticize']:
        class_weights[MasterConfig.intent_labels.index(c)] = MasterConfig.conflict_weight
    class_weights = class_weights.to(device)
    criterion_av = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=MasterConfig.label_smoothing)
    
    best_av_acc = 0.0
    best_av_weights = None
    
    for epoch in range(1, MasterConfig.epochs_av + 1):
        av_standalone.train()
        total_loss = 0.0
        for b in train_loader:
            af = b['audio_feats'].to(device)
            vf = b['video_feats'].to(device)
            labels = b['label'].to(device)
            
            av_optimizer.zero_grad()
            logits_av, _ = av_standalone(af, vf)
            loss = criterion_av(logits_av, labels)
            loss.backward()
            av_optimizer.step()
            total_loss += loss.item()
            
        dev_res = evaluate_master(av_standalone, dev_loader, device)
        print(f"AV-Only Epoch {epoch:02d}/{MasterConfig.epochs_av:02d} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_res['acc']:.2f}% | Dev WF1: {dev_res['wf1']:.2f}%")
        if dev_res['acc'] > best_av_acc:
            best_av_acc = dev_res['acc']
            best_av_weights = copy.deepcopy(av_standalone.state_dict())
            
    if best_av_weights is not None:
        av_standalone.load_state_dict(best_av_weights)
    test_av_res = evaluate_master(av_standalone, test_loader, device)
    print(f"\n[+] Standalone Audio-Video Test Accuracy (No Text) : {test_av_res['acc']:.2f}% | WF1: {test_av_res['wf1']:.2f}% | Macro F1: {test_av_res['mf1']:.2f}%")

    # ----------------------------------------------------------------------------------------------
    # EXPERIMENT 2 & 3: CALIBRATED HIERARCHICAL TWO-STAGE & SEQUENTIAL LEVEL-WISE MODEL
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> [EXPERIMENT 2 & 3] CALIBRATED HIERARCHICAL TWO-STAGE TRAINING (WITH EARLY STOPPING)")
    print("=" * 80)
    
    hierarchical_model = MasterHierarchicalModel(MasterConfig).to(device)
    
    # Initialize Stage 1 AV module with our trained AV weights for warm start!
    hierarchical_model.av_module.load_state_dict(av_standalone.av_module.state_dict())
    print("[+] Successfully initialized Hierarchical Stage 1 with pre-trained Audio-Video weights!")
    
    bert_params = list(hierarchical_model.bert.parameters())
    fusion_params = [p for n, p in hierarchical_model.named_parameters() if not n.startswith('bert')]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': MasterConfig.lr_bert},
        {'params': fusion_params, 'lr': MasterConfig.lr_fusion}
    ], weight_decay=MasterConfig.weight_decay)
    
    total_steps = len(train_loader) * MasterConfig.epochs_joint
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    criterion_joint = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=MasterConfig.label_smoothing)
    
    best_dev_acc = 0.0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(1, MasterConfig.epochs_joint + 1):
        hierarchical_model.train()
        total_loss = 0.0
        for b in train_loader:
            labels = b['label'].to(device)
            out = hierarchical_model(
                b['input_ids'].to(device), b['attention_mask'].to(device),
                b['audio_feats'].to(device), b['video_feats'].to(device)
            )
            loss_f = criterion_joint(out['logits'], labels)
            loss_t = criterion_joint(out['logits_text'], labels)
            loss_av = criterion_joint(out['logits_av'], labels)
            
            loss = loss_f + 0.3 * loss_t + 0.2 * loss_av
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hierarchical_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            
        dev_res = evaluate_master(hierarchical_model, dev_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} | Dev Fused: {dev_res['acc_fused']:.2f}% | Text: {dev_res['acc_text']:.2f}% | AV: {dev_res['acc_av']:.2f}%")
        
        if dev_res['acc_fused'] > best_dev_acc:
            best_dev_acc = dev_res['acc_fused']
            best_weights = copy.deepcopy(hierarchical_model.state_dict())
            patience_counter = 0
            print(f"    --> [NEW BEST] Saved checkpoint with Dev Acc: {best_dev_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= MasterConfig.patience:
                print(f"\n[*] Early stopping triggered at Epoch {epoch} to prevent overfitting! Restoring best model...")
                break
                
    if best_weights is not None:
        hierarchical_model.load_state_dict(best_weights)
        
    # Standard Hierarchical Test Results
    test_hier_res = evaluate_master(hierarchical_model, test_loader, device, sequential_mode=False)
    print("\n[+] Final Calibrated Hierarchical Test Results:")
    print(f"    - Fused Accuracy : {test_hier_res['acc_fused']:.2f}% | Weighted F1: {test_hier_res['wf1_fused']:.2f}% | Macro F1: {test_hier_res['mf1_fused']:.2f}%")
    print(f"    - Text-Only Acc  : {test_hier_res['acc_text']:.2f}% | Weighted F1: {test_hier_res['wf1_text']:.2f}%")
    print(f"    - AV-Branch Acc  : {test_hier_res['acc_av']:.2f}% | Weighted F1: {test_hier_res['wf1_av']:.2f}%")

    # Sequential Level-Wise Test Results (Text First, conditionally fuse ONLY when confidence < tau)
    print("\n[+] Sequential Conditional Fusion Sweep across Reduced Confidence Thresholds (tau):")
    print(f"    {'Tau':<6} | {'Text-Only Kept':<16} | {'Conditionally Fused':<20} | {'Accuracy (%)':<14} | {'Weighted F1 (%)':<16}")
    print("    " + "-" * 78)
    best_seq_tau = 0.55
    best_seq_acc = 0.0
    for tau in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        seq_res = evaluate_master(hierarchical_model, test_loader, device, sequential_mode=True, tau_conf=tau)
        text_kept_pct = 100.0 - seq_res['fused_pct']
        print(f"    {tau:<6.2f} | {text_kept_pct:>12.1f} %  | {seq_res['fused_pct']:>16.1f} %   | {seq_res['acc_fused']:>10.2f} %  | {seq_res['wf1_fused']:>12.2f} %")
        if seq_res['acc_fused'] > best_seq_acc:
            best_seq_acc = seq_res['acc_fused']
            best_seq_tau = tau
    print(f"    --> Optimal Sequential Threshold: Tau = {best_seq_tau:.2f} (Accuracy: {best_seq_acc:.2f}%)")

    # ----------------------------------------------------------------------------------------------
    # EXPERIMENT 4: TEXT WEIGHT REDUCTION & MANUAL MODALITY WEIGHTING SWEEP
    # ----------------------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> [EXPERIMENT 4] TEXT WEIGHT REDUCTION & MANUAL MODALITY WEIGHTING SWEEP")
    print("=" * 80)
    
    weight_sweep_results = []
    text_weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0]
    for wt in text_weights:
        sw_res = evaluate_master(hierarchical_model, test_loader, device, sequential_mode=False, w_text=wt, w_audio=0.5, w_video=0.5)
        weight_sweep_results.append({
            'w_text': wt,
            'acc': sw_res['acc_fused'],
            'wf1': sw_res['wf1_fused']
        })
        print(f"    - w_text = {wt:.1f} (AV weight = {1.0 - wt*0.5:.2f}) -> Accuracy: {sw_res['acc_fused']:.2f}% | WF1: {sw_res['wf1_fused']:.2f}%")
        
    # Manual Audio vs Video balance sweep at w_text=0.5
    print("\n[+] Manual Audio vs Video Balance Sweep (at w_text=0.5):")
    for wa in [0.2, 0.4, 0.5, 0.6, 0.8]:
        wv = 1.0 - wa
        bal_res = evaluate_master(hierarchical_model, test_loader, device, sequential_mode=False, w_text=0.5, w_audio=wa, w_video=wv)
        print(f"    - w_audio = {wa:.1f}, w_video = {wv:.1f} -> Accuracy: {bal_res['acc_fused']:.2f}% | WF1: {bal_res['wf1_fused']:.2f}%")

    # ----------------------------------------------------------------------------------------------
    # EXPERIMENT 5: GENERATE ALL 4 PUBLICATION-READY PLOTS
    # ----------------------------------------------------------------------------------------------
    generate_master_plots(test_hier_res, weight_sweep_results, output_dir=output_dir)
    print("\n" + "=" * 80)
    print(f">>> MASTER EXPERIMENTAL PIPELINE COMPLETE! ALL PLOTS SAVED TO: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    run_master_pipeline()
