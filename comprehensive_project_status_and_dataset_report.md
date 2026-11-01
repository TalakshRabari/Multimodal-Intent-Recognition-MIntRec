# Comprehensive Project Analysis & Dataset Status Report

**Project Title:** Multimodal Intent Recognition (Major Project)  
**Authors:** Talaksh Rabari & Swayam Patel (M.Tech Software Systems, Dhirubhai Ambani University)  
**Supervisor:** Dr. Shruti Bhilare  
**Date of Report:** September 2026  

---

## 1. Executive Summary & Where We Are

### 1.1 Project Overview & Trajectory
The objective of this major project is to advance **Multimodal Intent Recognition (MIR)** by combining **Text** (spoken words), **Audio** (prosody/acoustic cues), and **Visual** (facial expressions and body gestures) signals. The project builds upon the state-of-the-art **TCL-MAP** (Token-Level Contrastive Learning with Modality-Aware Prompting, AAAI 2024) architecture.

The technical development roadmap consists of four phases:
1. **Phase 1: Baseline Reproduction & Benchmarking (COMPLETE)**
   - Successfully reproduced TCL-MAP on the **MIntRec** benchmark with **73.48% Accuracy** and **73.02% Weighted F1**, closely matching the original paper (73.62% / 73.31%).
   - Benchmarked against **MISA**, **MulT**, and **MAG-BERT**. Prepared comprehensive IEEE report (`project_report.pdf`) and slide deck (`Multimodal Intent Recognition (1).pptx`).
2. **Phase 2: Prototype Memory & Prototype Loss Exploration (COMPLETE / ABLATED)**
   - Designed a learnable class prototype memory module (`PrototypeMemory`).
   - Investigated an explicit prototype attraction loss ($L_{\text{proto}} = \text{MSE}(\text{features}, \text{prototype}[\text{label}])$).
   - **Finding:** While $\lambda_{\text{proto}} = 0.05$ gained +0.75 points at 3 epochs, full 10-epoch training yielded no improvement over baseline (72.81% vs. 72.73% mean accuracy). It was recorded as a verified negative ablation and kept at $\lambda_{\text{proto}} = 0.0$.
3. **Phase 3: Gated Prototype Fusion & Hardware Scaling (CURRENT STATE)**
   - Transitioned from static/linear prototype fusion to **Gated Prototype Fusion** (`gate = sigmoid(Linear([original, memory]))`).
   - **Hardware Bottleneck Encountered:** The local machine has an **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)**. Backpropagation with batch size 16 triggers CUDA Out-Of-Memory (OOM).
   - **Transition to Kaggle T4 GPU (16 GB VRAM):** Required resolving compatibility issues between older codebase dependencies and modern Kaggle environments (PyTorch 2.x, Transformers 4.4x):
     - Legacy Hugging Face `AdamW` -> `torch.optim.AdamW`.
     - Deprecated `get_head_mask` -> fallback handling (`hasattr` & `try/except`).
     - Dynamic `get_extended_attention_mask` signature resolution.
4. **Phase 4: Advanced Dynamic Alignment & Uncertainty Fusion (NEXT STEPS)**
   - Evaluate Gated Prototype Fusion across 3 seeds on Kaggle T4.
   - Incorporate dynamic reliability weighting and uncertainty modeling, drawing inspiration from the **DUMIR** paper present in the workspace.

---

## 2. Exhaustive Workspace File Inventory & Analysis

Every file and directory in `c:\Users\talak\Major Poject_1` has been analyzed:

| Path / File | Type / Size | Role & Detailed Description |
| :--- | :--- | :--- |
| `project_report.pdf` | PDF (672 KB, 6 pages) | **Major Project Final/Interim Report:** Formatted in 2-column IEEE style by Talaksh Rabari & Swayam Patel, advised by Dr. Shruti Bhilare. Covers: MIR motivation, literature survey (MISA, TCL-MAP), dataset analysis (MIntRec, MELD-DA, IntentHQ), experimental reproduction results, and future research directions. |
| `Multimodal Intent Recognition (1).pptx` | Presentation (1.43 MB, 25 slides) | **Project Defense / Seminar Presentation:** Structured slide deck detailing background, multi-modality challenges, MISA architecture, TCL-MAP pipeline, prompt tuning, dataset profiles, reproduction tables, and proposed extensions. |
| `2312.14667v2.pdf` | PDF (950 KB, 9 pages) | **Primary Reference Paper (TCL-MAP):** *"Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition"* (AAAI 2024, Zhou et al., Tsinghua University). Proposes SBMA (Similarity-Based Modality Alignment) and Token-Level Contrastive Learning via prompt generation. |
| `2005.03545v3_copy.pdf` | PDF (10.9 MB, 10 pages) | **Secondary Reference Paper (MISA):** *"MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis"* (ACM MM 2020, Hazarika et al.). Deconstructs modalities into shared invariant and private specific subspaces via orthogonality and similarity constraints. |
| `DUMIR Dynamic Uncertainty-Aware Multimodal Intent Recognition.pdf` | PDF (3.32 MB, 14 pages) | **Advanced Reference Paper (DUMIR):** *"DUMIR: Dynamic Uncertainty-Aware Multimodal Intent Recognition"* (Pattern Recognition Journal, Gong et al., 2024). Introduces dynamic uncertainty estimation from feature sequence statistics and hierarchical cross-modal interaction to prevent noisy non-text modalities from corrupting predictions. |
| `TCL-MAP.zip` | Archive (1.88 MB) | Clean backup archive of the original TCL-MAP codebase prior to local modifications. |
| `datasets/` | Directory | Contains the core benchmark dataset `MIntRec` (TSVs and multimodal feature pickles). Detailed in Section 3. |
| `TCL-MAP/` | Directory | **Active Development Codebase:** Git-controlled repository with custom prototype memory modules, modified training manager, loss functions, configs, and logs. Detailed below. |
| `.agents/` | Directory | Clean/empty directory for AI agent state tracking. |

---

## 3. Detailed Dataset Analysis (`datasets/MIntRec`)

The dataset located at `datasets/MIntRec/` is the **MIntRec (Multimodal Intent Recognition)** benchmark, derived from conversational video clips of the TV sitcom *Friends*.

### 3.1 Data Splits & Sample Counts
- **Total Dataset Size:** 2,224 annotated multimodal dialogue samples.
  - `train.tsv`: 1,334 samples (60.0%)
  - `dev.tsv`: 445 samples (20.0%)
  - `test.tsv`: 445 samples (20.0%)
  - `all.tsv`: 2,224 samples (master compilation)

### 3.2 Label Space & Class Imbalance
MIntRec defines **20 fine-grained intent classes**. The class distribution across the full dataset exhibits pronounced real-world imbalance:

| Intent Class | Total Count | Train Split | Dev Split | Test Split | Category Characterization |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Complain** | 286 | 172 | 57 | 57 | Negative affective intention (high frequency) |
| **Inform** | 284 | 170 | 57 | 57 | Neutral informational statement |
| **Praise** | 213 | 127 | 43 | 43 | Positive social approval |
| **Apologise** | 136 | 82 | 27 | 27 | Social repair / reconciliation |
| **Thank** | 124 | 74 | 25 | 25 | Gratitude / politeness |
| **Advise** | 122 | 73 | 24 | 25 | Directive / recommendation |
| **Criticize** | 117 | 70 | 24 | 23 | Disapproval / reproach |
| **Arrange** | 110 | 66 | 22 | 22 | Planning / scheduling |
| **Introduce** | 105 | 63 | 21 | 21 | Introduction / initiation |
| **Care** | 95 | 57 | 19 | 19 | Empathy / concern |
| **Comfort** | 88 | 52 | 18 | 18 | Reassurance / consolation |
| **Leave** | 85 | 51 | 17 | 17 | Dismissal / departure |
| **Prevent** | 73 | 43 | 15 | 15 | Interruption / warning |
| **Taunt** | 62 | 38 | 12 | 12 | Teasing / provocation |
| **Greet** | 60 | 36 | 12 | 12 | Opening salutation |
| **Agree** | 59 | 35 | 12 | 12 | Concurrence / affirmation |
| **Flaunt** | 52 | 32 | 10 | 10 | Boasting / self-display |
| **Oppose** | 51 | 31 | 10 | 10 | Direct disagreement |
| **Ask for help** | 51 | 31 | 10 | 10 | Request for assistance |
| **Joke** | 51 | 31 | 10 | 10 | Humor / sarcasm |

*Imbalance ratio:* The most frequent class (`Complain`: 286) has over **5.6x** the samples of minority classes (`Joke`, `Ask for help`, `Oppose`: 51 each). This explains why Macro F1 (~67–70%) lags behind Weighted F1 and Accuracy (~73–74%).

### 3.3 Multimodal Features & Signal Dimensions

1. **Text Modality (`.tsv` files):**
   - Utterances transcribed from dialogue clips.
   - Text token length: Mean = 7.0 words, Median = 7.0 words, Min = 1 word, Max = 26 words.
   - Encoding backbone: `bert-base-uncased` (768 hidden dimensions).
   - Maximum text sequence length in config: 30 tokens.
   - Source seasons: S05 (1,272 clips), S04 (566 clips), S06 (386 clips).

2. **Video Modality (`video_feats.pkl` - 472.48 MB):**
   - 2,224 keys formatted as `Season_Episode_ClipID` (e.g., `S04_E10_464`).
   - Frame features extracted using **Swin Transformer** (1024 feature dimensions).
   - Shape per sample: `(num_frames, 1, 1024)` or `(num_frames, 1024)`.
   - Sequence length stats: Min = 13 frames, Max = 225 frames, Mean = 54.4 frames, Median = 50.0 frames.
   - Padded to max 230 frames during data loading.

3. **Audio Modality (`audio_feats.pkl` - 769.78 MB):**
   - 2,224 keys matching the video and text identifiers.
   - Frame-level acoustic representations extracted using **Wav2Vec 2.0** (768 feature dimensions).
   - Padded to max 480 frames during data loading.

---

## 4. Codebase Architecture & Modifications (`TCL-MAP/`)

### 4.1 Repository Structure
```
TCL-MAP/
├── configs/
│   ├── base.py                   # Parameter manager & argument parsing
│   ├── TCL_MAP_MIntRec.py        # MIntRec specific hyperparameters (batch size 16, lr 2e-5)
│   └── TCL_MAP_MELD.py           # MELD-DA config
├── data/
│   ├── base.py                   # DataManager coordinating text, video, audio loading
│   ├── MMDataset.py              # PyTorch Dataset wrapper
│   ├── mm_pre.py                 # Padding and preprocessing for video/audio pickles
│   └── text_pre.py               # Tokenization and prompt masking for BERT
├── methods/
│   └── TCL_MAP/
│       ├── AlignNets.py          # Similarity-Based Modality Alignment (SBMA)
│       ├── SubNets/              # Multihead cross-attention & transformer blocks
│       ├── modules/
│       │   └── prototype_memory.py # Learnable class prototypes with gated fusion
│       ├── loss.py               # SupConLoss, FocalLoss, PrototypeLoss
│       ├── manager.py            # Training/evaluation loop, optimizer, early stopping
│       └── model.py              # MAP_Model, Cons_Model, and full TCL_MAP module
├── logs/                         # 42 detailed experiment logs
├── outputs/                      # Saved checkpoints & prediction outputs
├── result_pic/                   # Confusion matrix plots & performance graphs
├── TCL_MAP_Project_Handoff.md    # Handoff notes detailing current progress & bugs
├── run.py                        # Main CLI entry point
└── requirements.txt              # Project dependencies
```

### 4.2 Key Architectural Modifications Made in Working Directory
1. **Prototype Memory Module (`methods/TCL_MAP/modules/prototype_memory.py`):**
   - Instantiates $C=20$ learnable prototype vectors in $\mathbb{R}^{768}$.
   - Performs $L_2$-normalized cosine similarity between pooled sample representations and prototypes.
   - Selects top-$k$ ($k=3$) nearest prototypes with softmax attention weighting.
   - **Gated Fusion:** Introduces a learned scalar gate $\alpha = \sigma(W [x; x_{\text{mem}}])$ to fuse original representations $x$ and memory features $x_{\text{mem}}$:
     $$\hat{x} = x + \alpha \cdot x_{\text{mem}}$$
2. **Loss Function Expansions (`methods/TCL_MAP/loss.py`):**
   - Added `PrototypeLoss`: MSE alignment to ground-truth class prototypes.
   - Added `FocalLoss`: $\alpha(1 - p_t)^\gamma \text{CE}(p, y)$ to address the severe 20-class imbalance.
   - Enhanced `SupConLoss` to support both supervised and unsupervised contrastive modes.
3. **Training Manager Updates (`methods/TCL_MAP/manager.py`):**
   - Replaced deprecated Hugging Face optimizer with `torch.optim.AdamW`.
   - Added diagnostic logging for first-batch activation/loss ranges.
   - Supported multi-loss weighting: $\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}}$.

---

## 5. Experimental History & Performance Records

A systematic review of all runs logged in `TCL-MAP/logs/` reveals key milestones:

| Experiment Series | Epochs | Batch | Loss Config | $\lambda_{\text{cons}}$ | $\lambda_{\text{proto}}$ | Accuracy (Mean) | Weighted F1 | Macro F1 | Key Findings / Insights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Official Paper** | 10 | 16 | CE | 1.0 | 0.0 | **73.62%** | **73.31%** | - | Benchmark target |
| **Local Baseline Reproduction** | 10 | 16 | CE | 1.0 | 0.0 | **73.48%** | **73.02%** | ~68.5% | Reproducibility confirmed across seeds 0, 1, 2 |
| **Prototype Loss Screening** | 3 | 16 | LabelSmooth | 1.0 | 0.00 | 70.04% | 68.37% | 63.36% | Baseline 3-epoch benchmark |
| **Prototype Loss Screening** | 3 | 16 | LabelSmooth | 1.0 | 0.05 | 70.79% | 68.62% | 63.30% | Early +0.75% gain observed |
| **Prototype Loss 10-Epoch Run** | 10 | 16 | LabelSmooth | 1.0 | 0.00 | 72.81% | 72.03% | 68.23% | Full baseline |
| **Prototype Loss 10-Epoch Run** | 10 | 16 | LabelSmooth | 1.0 | 0.05 | 72.73% | 72.13% | 68.18% | **Negative Ablation:** No 10-epoch gain; $\lambda_{\text{proto}} \to 0$ |
| **Focal Loss Exploration** | 10 | 16 | Focal ($\gamma=2$) | 1.0 (sup) | 0.00 | 71.46% | 70.85% | 67.06% | Focal loss underperformed standard CE/LabelSmooth |
| **Gated Prototype Fusion** | 3 | 16 | LabelSmooth | 1.0 | 0.0 | **Pending** | - | - | CUDA OOM on local 4GB GPU; ready for Kaggle T4 |

---

## 6. Kaggle T4 GPU Environment & Execution Guide

### 6.1 Hardware Advantage
- **Local RTX 3050 Laptop:** 4 GB VRAM $\implies$ OOM during backward pass at batch size 16. Epoch duration: ~4.5 minutes.
- **Kaggle T4 GPU:** 16 GB GDDR6 VRAM $\implies$ Easily handles batch size 16 (or 32), with fast FP16/mixed precision support. Epoch duration: ~45–60 seconds (~5x speedup).

### 6.2 Kaggle Setup & Compatibility Fixes
When executing on Kaggle (Python 3.12, modern Transformers), verify the four compatibility fixes:

1. **Optimizer:** In `methods/TCL_MAP/manager.py`:
   ```python
   from torch.optim import AdamW
   optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-6)
   ```
2. **Head Mask Fallback:** In `methods/TCL_MAP/model.py` (lines 237–244 and 519–525), ensure the call is guarded:
   ```python
   if hasattr(self, 'get_head_mask'):
       try:
           head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
       except Exception:
           head_mask = None
   else:
       head_mask = None
   ```
3. **Tied Weights Attribute:** In `MAP_Model` and `Cons_Model`:
   ```python
   self.all_tied_weights_keys = {}
   ```
4. **Extended Attention Mask:** Safe dual-mode call:
   ```python
   try:
       extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
   except TypeError:
       extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
   ```

### 6.3 Ready-to-Run Kaggle Execution Cell
In your Kaggle Notebook, set the accelerator to **GPU T4 x1** and run:

```python
import os
import subprocess
from pathlib import Path

# 1. Copy code to writable working directory
!rm -rf /kaggle/working/TCL-MAP
!cp -r /kaggle/input/tcl-map-source/TCL-MAP /kaggle/working/TCL-MAP

# 2. Locate dataset path
train_file = next(Path('/kaggle/input').rglob('train.tsv'))
data_path = str(train_file.parent.parent)
print(f"Detected dataset root path: {data_path}")

# 3. Screen Gated Prototype Fusion across seeds 0, 1, 2 (3 epochs)
for seed in [0, 1, 2]:
    print(f"\n========== RUNNING SEED {seed} (3 Epochs Screening) ==========\n")
    subprocess.run([
        'python', 'run.py',
        '--train',
        '--seed', str(seed),
        '--loss_type', 'labelsmooth',
        '--lambda_cons', '1.0',
        '--lambda_proto', '0.0',
        '--num_train_epochs', '3',
        '--data_path', data_path
    ], cwd='/kaggle/working/TCL-MAP', check=True)
```

---

## 7. Recommended Next Steps & Research Trajectory

1. **Immediate Execution:** Run the 3-epoch screening of Gated Prototype Fusion on Kaggle T4.
   - Compare the 3-seed mean against the ungated baseline (70.04% ACC).
   - If mean $> 70.04\%$, train for 10 epochs (baseline is 72.81%).
2. **Dynamic Reliability / Uncertainty Fusion (DUMIR Adaptation):**
   - Rather than relying on a static scalar gate, implement sequence-level variance/reliability scoring as proposed in DUMIR.
   - Modulate acoustic and visual feature contributions before prompt generation so that ambiguous/noisy non-text frames do not degrade BERT's representations.
3. **Graph Cross-Attention:**
   - Incorporate graph message-passing across modal tokens to capture higher-order inter-modality dependencies.
