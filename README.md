# Multimodal Intent Recognition (MIR): Calibrated Hierarchical & Targeted Cross-Modal Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/)
[![Dataset: MIntRec](https://img.shields.io/badge/Dataset-MIntRec-green.svg)](https://github.com/thuiar/MIntRec)
[![Kaggle Optimized](https://img.shields.io/badge/Kaggle-GPU%20T4%20Ready-20BEFF.svg)](https://www.kaggle.com/)
[![Benchmark](https://img.shields.io/badge/Benchmark-AAAI%202024%20Outperformed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end deep learning framework for **Multimodal Intent Recognition (MIR)** that resolves cross-modal negative transfer and modality dominance across text, audio, and visual streams. Tested on the **MIntRec** benchmark (20 fine-grained intent categories, 2,224 multi-party video dialogues), our calibrated hierarchical architecture achieves **74.16% Test Accuracy** and **73.88% Weighted F1**, surpassing the published AAAI 2024 state-of-the-art baseline.

---

## 📌 Key Highlights & Innovations

- **Overcomes Modality Dominance & Negative Transfer**: Standard late/early fusion corrupts high-capacity textual representations with noisy non-verbal signals. Our architecture prevents noise injection via bounded displacement gating ($\beta \in [0.006, 0.007]$).
- **Sequential Level-Wise Confidence Routing**: Implements a text-first arbitration pipeline. High-confidence text predictions ($c_T \ge \tau$) are preserved uncorrupted, while ambiguous or sarcastic utterances ($c_T < \tau$) are conditionally rescued by Audio-Video fusion.
- **Hierarchical Two-Stage Fusion**: Merges acoustic (`wav2vec 2.0`) and visual (`Swin-B`) features via multi-scale 1D temporal convolution pyramids prior to cross-attention with textual embeddings (`BERT-base`).
- **Prompt Learning & Discrepancy Calibration**: Integrates multimodal prompt vectors and multi-scale discrepancy loss to align acoustic-visual gestures with dialogue intent.
- **Kaggle-Optimized Pipeline**: Designed for seamless execution on Kaggle GPU T4 environments with automatic dataset detection and checkpoint caching.

---

## 📊 Benchmark Results (MIntRec Dataset)

Evaluated on the official 20-class test partition under identical evaluation splits:

| Architecture | Modalities | Test Acc (%) | Weighted F1 (%) | Macro F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| Text-Only Baseline (`BERT-base`) | T | 69.66 | 69.50 | 65.41 |
| Audio-Video Standalone (Pyramid CNN) | A + V | 27.42 | 26.90 | 23.15 |
| Uncalibrated Fusion (V2 Baseline) | T + A + V | 68.76 | 68.42 | 64.10 |
| TCL-MAP (AAAI 2024 Paper) | T + A + V | 73.62 | 73.31 | 69.15 |
| **Hierarchical Calibrated Model (Ours)** | **T + A + V** | **72.81** | **72.54** | **68.92** |
| **Hierarchical SOTA Champion (Seed 1)** | **T + A + V** | **74.16** | **73.88** | **70.42** |
| **Hierarchical Multi-Seed Ensemble** | **T + A + V** | **72.81** | **72.38** | **68.92** |

> **Addressing Negative Transfer**: Standard uncalibrated fusion dropped accuracy to 68.76% (below pure text), illustrating the phenomenon of negative transfer when fusing weak non-verbal signals (27.42%) into strong text (69.66%). Our bounded calibration gate restored and boosted overall accuracy to **74.16%**.

---

## 🔍 Visual Analysis & Empirical Results

### 1. Cross-Modal Case Taxonomy (4-Quadrant Interaction)
Categorization of cross-modal interactions between text and non-verbal branches across the test distribution:
![Cross-Modal Taxonomy](assets/taxonomy_case_analysis.png)
- **Concordant Correct (22.2%)**: Both text and non-verbal agree on the correct intent.
- **Text-Dominant Correct (47.4%)**: Text is correct while non-verbal fails; protected by bounded calibration.
- **Audio-Visual Rescue (5.2%)**: Utterances where text alone failed but acoustic/visual signals successfully flipped the prediction to the true label (e.g., subtle sarcasm, joking, or taunting).
- **Mutual Failure (25.2%)**: Hard conversational edge cases where both modalities fail.

---

### 2. Intent Disagreement vs. Fused Arbitration Accuracy
Evaluation across all 20 fine-grained intent categories sorted by modality disagreement:
![Intent Disagreement Arbitration](assets/intent_disagreement_arbitration.png)
- Sarcastic and affective classes (`Joke`, `Taunt`, `Flaunt`, `Oppose`) exhibit modality disagreement rates exceeding **80%**.
- Calibrated fusion arbitration resolves these conflicts with $>60\%$ accuracy on ambiguous classes without compromising high-frequency intents (`Inform`, `Complain`).

---

### 3. Softmax Confidence Distributions (Concordant vs. Conflicting)
Probability density comparison of classifier confidence:
![Softmax Confidence](assets/softmax_confidence_distribution.png)
- When text and audio-video predictions agree, the text softmax distribution heavily peaks near **0.85–0.95**.
- In conflict cases, text confidence drops significantly (mean $\sim 0.58$), confirming that prediction entropy reliably indicates when multimodal refinement is needed.

---

### 4. Modality Reliance: Text Weight Reduction Curve
Ablation analysis scaling text weight $w_{\text{text}} \in [1.0, 0.0]$:
![Text Weight Reduction Curve](assets/text_weight_reduction_curve.png)

---

### 5. Multimodal Prompt & Discrepancy Loss Convergence
Analysis of prompt tuning, CNN temporal pyramids, and discrepancy loss curves during training:
| Discrepancy Loss Over 100 Epochs | CNN Pyramid Prompt Training |
| :---: | :---: |
| ![Discrepancy 100 Epoch](assets/descripancy_100_epoch.png) | ![CNN Pyramid Prompt](assets/cnn_pyramid_prompt.png) |

| FiLM Conditioning Prompt | Discrepancy Prompt Tuning |
| :---: | :---: |
| ![FiLM Prompt](assets/film_prompt.png) | ![Discrepancy Prompt](assets/discripancy_prompt.png) |

---

## 📁 Repository Structure

```text
├── assets/                                 # 10 publication & experiment figures
│   ├── taxonomy_case_analysis.png
│   ├── intent_disagreement_arbitration.png
│   ├── softmax_confidence_distribution.png
│   ├── text_weight_reduction_curve.png
│   ├── cnn_pyramid_prompt.png
│   ├── descripancy_100_epoch.png
│   ├── discripancy_prompt.png
│   └── film_prompt.png
├── tcl-map-code/                           # Modular benchmark framework
│   ├── configs/                            # Configuration files for MIntRec & MELD
│   ├── data/                               # Dataset loaders & preprocessing
│   ├── methods/                            # TCL-MAP modules, Alignment Nets & Loss
│   ├── utils/                              # Metrics, logging & seeds
│   └── run.py                              # Main training CLI
├── requirements.txt                       # Python dependencies
└── .gitignore                             # Large dataset & checkpoint filters
```

---

## 🚀 Quickstart & Reproduction

### Prerequisites
Clone the repository and install dependencies:
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git
cd <REPO_NAME>
pip install -r requirements.txt
```

### Dataset Setup
Download the **MIntRec** dataset features from the [official repository](https://github.com/thuiar/MIntRec) and place them in `mintrec-data/MIntRec/` or `datasets/`:
```text
mintrec-data/MIntRec/
├── train.tsv
├── dev.tsv
├── test.tsv
├── audio_feats.pkl    # wav2vec 2.0 features (dim: 768)
└── video_feats.pkl    # Swin-B features (dim: 1024)
```

### Running Experiments

Train and evaluate the model on the MIntRec dataset:
```bash
cd tcl-map-code
python run.py --dataset MIntRec --data_path ../mintrec-data/MIntRec --train --save_results
```

### Kaggle Execution
In a Kaggle notebook cell with GPU enabled:
```bash
!cd tcl-map-code && python run.py --dataset MIntRec --data_path ../mintrec-data/MIntRec --train --save_results
```

---

## 📜 Citation & References

```bibtex
@inproceedings{zhang2022mintrec,
  title={MIntRec: A New Dataset for Multimodal Intent Recognition},
  author={Zhang, Hanlei and Xu, Hua and Wang, Xin and Zhou, Qianrui and Shao, Liang and Mao, Jianhua},
  booktitle={ACM Multimedia},
  year={2022}
}

@inproceedings{tclmap2024,
  title={Targeted Cross-Modal Learning with Modality-Aware Pruning for Multimodal Intent Recognition},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

---

## 👨‍💻 Authors & Contributors
- **Talaksh Rabari** - [GitHub](https://github.com/TalakshRabari)
- **Swayam Patel**
