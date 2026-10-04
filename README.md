# Learning Across Layers and Time for Deepfake Detection

**Learning Across Layers and Time for Deepfake Detection** proposes a two-stage video deepfake detector built on DINOv3-Large.

## Table of Contents

- [Introduction](#introduction)
- [Repository Layout](#repository-layout)
- [Quick Start](#quick-start)
  - [1. Installation](#1-installation)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Stage 1: Frame Training](#3-stage-1-frame-training)
  - [4. Stage 2: Video Training](#4-stage-2-video-training)
- [Method Overview](#method-overview)
- [Citation](#citation)

---

## Introduction

The method learns forensic cues at multiple DINOv3 depths and then models how those cues evolve across video frames.

- **Stage 1** adapts a pretrained DINOv3-Large backbone with LoRA. Four Layer Token Heads use CLS, register, and patch tokens from transformer blocks 20–23.
- **Stage 2** freezes the frame model, retrieves real-video references from a k-nearest-neighbor memory bank, and processes four frame-feature sequences with temporal transformers. A pooled frame-logit residual is combined with the temporal representation for video classification.

## Repository Layout

```text
.
├── augmentations.py       # image loading, normalization, and training augmentations
├── frame_model.py         # DINOv3-Large, LoRA, and Layer Token Heads
├── video_model.py         # temporal modules and real-video memory bank
├── train_stage1.py        # frame-level training
├── train_stage2.py        # video-level training
├── requirements.txt
└── README.md
```

Datasets, pretrained weights, and generated checkpoints are kept outside the source tree.

## Quick Start

### 1. Installation

Install a PyTorch build compatible with your CUDA setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The DINOv3-Large pretrained weights are loaded through `timm` when the frame model is initialized. FAISS is optional; without it, memory-bank search uses the PyTorch implementation.

### 2. Data Preparation

Prepare CSV manifests with these columns:

- `sample_dir`: relative path to a frame directory containing `image.png`
- `label`: `0` for real and `1` for fake

For video training, include one row per frame. Frame directory names should end in `_frame_<number>` or `_f<number>` so the loader can group frames by video. The example commands below assume the manifest and image roots are stored under `data/`; replace them with your local paths.

The paper trains on the FaceForensics++ c23 protocol and samples 32 frames uniformly per video for Stage 2.

### 3. Stage 1: Frame Training

```bash
python train_stage1.py --manifest data/ffpp/manifest.csv --root_dir data/ffpp --save_root checkpoints/stage1 --epochs 50
```

Stage 1 saves `best.pth` and `latest.pth` under the selected `--save_root`.

### 4. Stage 2: Video Training

```bash
python train_stage2.py --load_from checkpoints/stage1/best.pth --manifest data/ffpp/manifest.csv --root_dir data/ffpp --save_root checkpoints/stage2 --num_frames 32 --batch_size 10 --use_memory_bank
```

Stage 2 freezes the frame model and trains the temporal transformers and video classifier. The real-video memory bank is constructed from real training videos. The selected checkpoint is saved as `best.pth` under `--save_root`.

## Method Overview

The Layer Token Head combines three complementary signals at each of four backbone depths: the CLS token, mean-pooled register tokens, and multi-head attention-pooled patch tokens. Stage 1 applies binary classification, supervised contrastive, and multi-similarity objectives, with the final tapped layer receiving primary supervision.

For Stage 2, each tapped layer supplies a sequence of frame CLS features. A memory bank stores pooled real-video features for each stream; cosine nearest neighbors are combined with similarity-weighted averaging and injected through a learnable gate. A dedicated temporal transformer processes each stream. The four video-token outputs are concatenated with the mean deepest-layer frame logits and classified at video level.

## Citation

```bibtex
TODO
```


#
