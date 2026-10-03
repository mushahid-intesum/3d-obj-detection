# MonoMH-Enhanced — How to Run

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation & Testing](#evaluation--testing)
- [Enhancement Modules](#enhancement-modules)
- [Running Ablation Experiments](#running-ablation-experiments)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- NVIDIA GPU with ≥ 8 GB VRAM (RTX 2080 or better recommended)
- 16 GB+ system RAM

### Software

- Python 3.7+
- PyTorch 1.7+ with CUDA support
- torchvision (matching PyTorch version)

### Install Dependencies

```bash
pip install torch torchvision
pip install pyyaml tqdm opencv-python matplotlib numpy wandb
```

> [!NOTE]
> Weights & Biases (`wandb`) is used for training logging. Run `wandb login` before your first training session, or set `WANDB_MODE=offline` to skip.

---

## Project Structure

```
MonoMH-Enhanced/
├── lib/
│   ├── kitti.yaml                  # Main configuration file
│   ├── datasets/
│   │   ├── kitti.py                # KITTI dataset loader
│   │   ├── kitti_utils.py          # Calibration, 3D box utilities
│   │   └── utils.py                # Angle encoding, heatmap utils
│   ├── models/
│   │   └── MonoMH.py               # Model architecture
│   ├── losses/
│   │   └── loss_function.py        # SoftBoM loss + HTL
│   ├── helpers/
│   │   ├── trainer_helper.py       # Training loop
│   │   ├── tester_helper.py        # Evaluation loop
│   │   ├── decode_helper.py        # Detection decoding + hypothesis generation
│   │   ├── model_helper.py         # Model builder
│   │   ├── dataloader_helper.py    # Dataloader builder
│   │   └── save_helper.py          # Checkpoint save/load
│   └── modules/                    # Enhancement modules (all config-gated)
│       ├── prototype_filter.py     # A1: Prototype-based hypothesis filtering
│       ├── intrinsic_conditioner.py# B2: Focal-length-aware threshold
│       ├── depth_prior.py          # C2: Depth prior warm-start
│       └── early_exit.py           # D1: Hypothesis early exit
├── tools/
│   ├── train_val.py                # Main entry point (train/eval/test)
│   ├── train.sh                    # Training launcher
│   ├── eval.sh                     # Validation launcher
│   ├── test.sh                     # Test set launcher
│   ├── eval.py                     # KITTI AP evaluation
│   ├── precompute_depth_priors.py  # Offline depth map generation (for C2)
│   ├── run_ablations.py            # Ablation experiment runner
│   └── analyze_ablations.py        # Post-hoc ablation analysis
└── outputs/                        # Training outputs (auto-created)
```

---

## Data Preparation

### 1. Download KITTI 3D Object Detection Dataset

Download from [KITTI website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d):

- Left color images (`image_2`)
- Camera calibration matrices (`calib`)
- Training labels (`label_2`)

### 2. Organize Directory

```
/path/to/kitti/
├── training/
│   ├── image_2/        # 7481 PNG images
│   ├── calib/          # 7481 calibration files
│   ├── label_2/        # 7481 label files
│   └── depth_prior/    # (optional) precomputed depth maps for C2
└── testing/
    ├── image_2/
    └── calib/
```

### 3. Set Data Path

Edit `lib/kitti.yaml`:

```yaml
dataset:
  root_dir: '/path/to/kitti'
```

Or pass it via the ablation runner's `--data_dir` argument.

### 4. Precompute Depth Priors (optional — only needed for C2)

```bash
python tools/precompute_depth_priors.py \
    --data_dir /path/to/kitti/training \
    --output_dir /path/to/kitti/training/depth_prior \
    --device cuda
```

This generates a `.npy` depth map for each training image. The script tries Depth Anything V2 first, falls back to MiDaS, then to a simple gradient proxy.

> [!TIP]
> On a V100 GPU, precomputation takes ~20 minutes for the full KITTI training set. You only need to do this once.

---

## Configuration

All settings are in `lib/kitti.yaml`. Key sections:

### Dataset

| Key | Default | Description |
|-----|---------|-------------|
| `root_dir` | `/your/data/path/KITTIDataset` | **Must set** — Path to KITTI root |
| `batch_size` | 16 | Training batch size |
| `num_workers` | 8 | Dataloader workers |
| `eval_cls` | `['Car', 'Pedestrian', 'Cyclist']` | Classes to evaluate |

### Training

| Key | Default | Description |
|-----|---------|-------------|
| `max_epoch` | 200 | Total training epochs |
| `eval_start` | 10 | First evaluation epoch |
| `eval_frequency` | 10 | Evaluate every N epochs |
| `disp_frequency` | 100 | Print loss every N batches |

### Optimizer

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | 0.00125 | Learning rate |
| `weight_decay` | 0.00001 | Weight decay |

### Enhancements

All enhancement modules are **disabled by default**. Set `enabled: true` to activate. See [Enhancement Modules](#enhancement-modules) for details.

---

## Training

### Baseline Training (no enhancements)

```bash
# Using shell script
CUDA_VISIBLE_DEVICES=0 bash tools/train.sh \
    --config lib/kitti.yaml \
    --work-date exp_baseline \
    --work-dir run1

# Or directly
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py \
    --config lib/kitti.yaml \
    --work-date exp_baseline \
    --work-dir run1 \
    --save-path outputs/
```

### Training with Enhancements

1. Edit `lib/kitti.yaml` and set `enabled: true` for desired modules
2. Run training as above

Example — enable all enhancements:

```yaml
enhancements:
  prototype_filter:
    enabled: true       # A1
  intrinsic_conditioner:
    enabled: true       # B2
  depth_prior:
    enabled: true       # C2 (requires precomputed depth priors)
  early_exit:
    enabled: true       # D1
```

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py \
    --config lib/kitti.yaml \
    --work-date exp_full \
    --work-dir run1
```

### Output Structure

Training produces:

```
outputs/<work-date>/<work-dir>/
├── bsz_16_lr_0.00125_adam_train.log   # Training log
├── checkpoints/
│   └── checkpoint_epoch_XXX           # Best model checkpoint
├── epoch_XXX/
│   └── data/                          # Per-epoch detection results
└── wandb/                             # W&B offline logs
```

---

## Evaluation & Testing

### Evaluate on Validation Set

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py \
    --config lib/kitti.yaml \
    -e \
    --work-date eval_run
```

> [!IMPORTANT]
> Set `tester.resume_model` in `kitti.yaml` to the checkpoint path before running evaluation:
> ```yaml
> tester:
>   resume_model: 'outputs/exp_baseline/run1/checkpoints/checkpoint_epoch_200'
>   out_dir: 'outputs/eval_results'
> ```

### Evaluate on Test Set

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_val.py \
    --config lib/kitti.yaml \
    -t \
    --work-date test_run
```

Test results are saved to `tester.out_dir` and can be submitted to the [KITTI benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

### Evaluation Metrics

The evaluation reports AP₃D (Average Precision for 3D detection) at IoU threshold 0.70 for three difficulty levels:

| Level | Description |
|-------|-------------|
| **Easy** | Fully visible, large bounding boxes |
| **Moderate** | Partially occluded, medium bounding boxes |
| **Hard** | Heavily occluded, small bounding boxes |

---

## Enhancement Modules

### A1: Prototype-Based Hypothesis Filtering

**Source:** Adapted from MonoSAOD

Replaces the hardcoded confidence threshold for hypothesis generation with a learned prototype bank. During training, ground-truth RoI features are accumulated into a prototype bank via EMA. During inference, generated depth hypotheses are filtered by depth reliability.

```yaml
prototype_filter:
  enabled: true
  num_prototypes: 192         # max prototype centroids
  members_per_proto: 512      # FIFO buffer per prototype
  similarity_threshold: 0.85  # cosine similarity threshold
  depth_reliability_threshold: 0.3  # min depth reliability
  ema_alpha: 0.005            # EMA update rate
  feature_dim: 64             # backbone feature dimension
  warmup_epochs: 10           # delay before bank is used
```

**Expected effect:** Improved AP₃D Hard (+0.5–1.5%) by filtering unreliable hypotheses.

### B2: Intrinsic-Conditioned Hypothesis Diversity

**Source:** Adapted from MonoIA

A lightweight MLP that maps the camera focal length to a per-image confidence threshold, replacing the fixed `0.75` threshold in hypothesis generation. Cameras with shorter focal lengths (more depth ambiguity) get lower thresholds, enabling more hypotheses.

```yaml
intrinsic_conditioner:
  enabled: true
  base_threshold: 0.75       # baseline threshold
  modulation_range: 0.3      # max shift range
  hidden_dim: 32              # MLP hidden size (~97 params)
```

**Expected effect:** Better cross-camera generalization; adaptive hypothesis count.

### C2: Depth Prior Warm-Start

**Source:** Adapted from FF-VIO-Init

Uses precomputed relative depth maps (from Depth Anything V2 or similar) to warm-start the depth head during early training. A learnable affine alignment (`scale`, `shift`) maps relative depth to metric depth. The prior loss weight decays linearly to zero over `warmstart_epochs`.

```yaml
depth_prior:
  enabled: true
  prior_dir: 'depth_prior'    # relative to kitti/training/
  warmstart_epochs: 50        # decay period
  prior_loss_weight: 1.0      # initial loss weight
```

> [!IMPORTANT]
> Requires precomputed depth priors. Run `tools/precompute_depth_priors.py` first.

**Expected effect:** Faster convergence (10–20 fewer epochs to reach peak AP₃D).

### D1: Hypothesis Early Exit

**Source:** Novel contribution

An `AmbiguityPredictor` head classifies each RoI as EASY, MEDIUM, or HARD based on depth variance. During inference, EASY RoIs skip hypothesis generation entirely, MEDIUM RoIs evaluate 3 center regions, and HARD RoIs use all 9 regions.

```yaml
early_exit:
  enabled: true
  easy_threshold: 0.2        # ambiguity < 0.2 → skip regions
  hard_threshold: 0.7        # ambiguity ≥ 0.7 → all 9 regions
  hidden_dim: 64              # predictor hidden size (~4.5K params)
  loss_weight: 0.1            # BCE loss weight for training
```

**Expected effect:** 50–70% reduction in hypothesis region evaluations with < 0.5% AP₃D drop.

---

## Running Ablation Experiments

The ablation runner automates the full experimental matrix:

```
Exp  A1   B2   C2   D1   Purpose
────────────────────────────────────────────
 0   off  off  off  off  Baseline reproduction
 1   ON   off  off  off  Prototype filter effect
 2   off  ON   off  off  Adaptive threshold effect
 3   off  off  ON   off  Convergence acceleration
 4   off  off  off  ON   Inference speedup
 5   ON   off  off  ON   Filter + speed combo
 6   ON   ON   ON   ON   Full integration
```

### Quick Sanity Check (30 epochs)

```bash
python tools/run_ablations.py \
    --data_dir /path/to/kitti \
    --gpu 0 \
    --quick
```

### Full Ablation (200 epochs)

```bash
python tools/run_ablations.py \
    --data_dir /path/to/kitti \
    --gpu 0
```

### Run Specific Experiments Only

```bash
python tools/run_ablations.py \
    --data_dir /path/to/kitti \
    --gpu 0 \
    --exps 0 3 6
```

### Dry Run (generate configs only)

```bash
python tools/run_ablations.py \
    --data_dir /path/to/kitti \
    --dry_run
```

### Analyze Results

After training completes:

```bash
python tools/analyze_ablations.py \
    --results_dir outputs/ablation
```

This generates:
- `ablation_analysis.md` — Full comparison report with AP₃D tables
- `csv/` — Per-experiment AP₃D trajectories for external plotting

> [!TIP]
> The ablation runner automatically disables C2 if depth priors haven't been precomputed, so you can always run the full matrix safely.

---

## Troubleshooting

### `torchvision.ops.nms` import error

**Symptom:** `ImportError: cannot import name 'nms' from 'torchvision.ops'`

**Fix:** Ensure your PyTorch and torchvision versions match:

```bash
pip install torch==1.13.1 torchvision==0.14.1
```

### CUDA out of memory

**Fix:** Reduce batch size in `lib/kitti.yaml`:

```yaml
dataset:
  batch_size: 8  # or 4
```

### wandb login prompt blocks training

**Fix:** Disable wandb:

```bash
WANDB_MODE=offline python tools/train_val.py --config lib/kitti.yaml ...
```

### Depth priors not found (C2)

**Symptom:** C2 enabled but training shows `depth_prior_loss: 0.0000` every epoch.

**Fix:** Precompute priors first:

```bash
python tools/precompute_depth_priors.py \
    --data_dir /path/to/kitti/training
```

Verify they exist:

```bash
ls /path/to/kitti/training/depth_prior/ | head
# Should show: 000000.npy, 000001.npy, ...
```

### Evaluation shows all zeros

**Symptom:** AP₃D Easy/Mod/Hard all report 0.00.

**Fix:** Ensure `dataset.root_dir` points to the directory **containing** `training/` (not `training/` itself):

```yaml
# Correct:
root_dir: '/path/to/kitti'

# Wrong:
root_dir: '/path/to/kitti/training'
```

### Enhancement has no effect

**Symptom:** Enabling an enhancement doesn't change results.

**Checklist:**
1. Verify `enabled: true` in the config YAML
2. Check the training log for module initialization messages (e.g., `[A1] PrototypeBank initialized`)
3. For A1: ensure `warmup_epochs` has passed
4. For C2: ensure depth prior `.npy` files exist
5. For D1: the amb_loss should appear in training loss display
