# PSRN: Pose-Based Two-Stream Relational Networks for Action Recognition

Implementation of "Pose-Based Two-Stream Relational Networks for Action Recognition in Videos" (arXiv:1805.08484).

## Overview

PSRN combines temporal pose dynamics with spatial object features for action recognition in videos:

1. **Temporal pose stream**: Multi-person 2D pose estimation with attention-based person selection, encoded by BiLSTMs for position and velocity sequences
2. **Spatial object stream**: VGG16 features from a randomly sampled frame (7×7×512 feature map)
3. **Relational Network**: Learns relations between pose features and object features for fusion

**Results**: Sub-JHMDB 80.2%, PennAction 98.1% (from paper)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
psrn/
  configs.py          # Configuration dataclasses
  train.py            # Training script with 3-stage schedule
  eval.py             # Evaluation script
  models/             # Model components
    parts_mlp.py      # Body-part encoder
    pose_selector.py  # Attention-based person selector
    pose_stream.py    # BiLSTM encoders for position/velocity
    object_stream.py  # VGG16 feature extractor
    relational.py     # Relational Network
    psrn.py           # Main PSRN model
  data/               # Data loading and preprocessing
    datasets.py       # PSRNDataset class
    preprocessing.py  # Pose and image preprocessing
  utils/              # Utilities
    metrics.py        # Loss and accuracy computation
    schedulers.py     # Learning rate schedulers
    logging.py        # Logging setup
```

## Data Preparation

1. **Extract poses**: Run a multi-person 2D pose estimator (e.g., OpenPose, PAF) over all video frames
2. **Save pose JSON**: Per-frame JSON format: `{"people": [{"keypoints": [[x, y, conf], ...]}, ...]}`
3. **Organize data**: 
   - For Sub-JHMDB: `data/split_{1,2,3}/videos/{action}/{video_name}/pose_*.json`
   - For PennAction: Adapt structure accordingly

The preprocessing pipeline will:
- Fill missing joints with (0, 0)
- Pad to max N persons per video
- Normalize keypoints to [0, 1]
- Split into 5 body parts (head, left/right arm, left/right leg)

## Training

### 3-Stage Training Schedule

From Sec. 3.5 of the paper:

**Stage 1**: Train pose stream only (lr=1e-4, halve after 78k iterations)
```bash
python -m psrn.train --data_root data/sub-jhmdb --split 1 --num_classes 12 --stage 1
```

**Stage 2**: Freeze pose; train VGG16 + RN (warmup 1e-6→1e-4 over 2k steps, halve after 28k)
```bash
python -m psrn.train --data_root data/sub-jhmdb --split 1 --num_classes 12 --stage 2
```

**Stage 3**: End-to-end fine-tuning
```bash
python -m psrn.train --data_root data/sub-jhmdb --split 1 --num_classes 12 --stage 3
```

Or train all stages sequentially:
```bash
python -m psrn.train --data_root data/sub-jhmdb --split 1 --num_classes 12
```

### Configuration

Key hyperparameters (from paper):
- **Batch size**: 16
- **Frames per video**: 10 (uniformly sampled)
- **Learning rates**: Stage 1/3: 1e-4, Stage 2: warmup 1e-6→1e-4
- **Weight decay**: 4e-5
- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Gradient clipping**: 5.0

## Evaluation

```bash
python -m psrn.eval --data_root data/sub-jhmdb --split 1 --num_classes 12 --checkpoint checkpoints/best.pt
```

For stability, the evaluation averages predictions over 10 random samples per video (configurable via `--num_samples`).

## Model Architecture

### Pose Stream
1. **Body-part encoding**: Split 14 keypoints into 5 parts → MLP each to 100-D → concat → 500-D
2. **Attention selection**: At each timestep, compute attention over N persons → attended pose `l_t`
3. **BiLSTM encoding**: 
   - Position: BiLSTM over `l_1..l_T` → `h_T^L` (512-D)
   - Velocity: BiLSTM over `V_t = l_{t+1} - l_t` → `h_T^V` (512-D)

### Object Stream
- VGG16 on one randomly sampled frame → 7×7×512 feature map → 49×512 object vectors

### Relational Network
- For each object `x_i`: `g_θ([h_T^L, h_T^V, x_i])` (4-layer MLP, 512-D per layer)
- Sum over objects → `f_φ(·)` (2-layer MLP, 512-D per layer) → relation feature `R` (512-D)

### Classification
Three heads: `Linear(512, C)` for position, velocity, and relational streams.

Loss: `L_total = L_pos + L_vel + L_rel + 4e-5‖θ‖²`

## Implementation Details

- **Pose normalization**: Keypoints normalized to [0, 1] by dividing by frame width/height
- **Pose filling**: Missing joints → (0, 0); frames with <N persons → virtual persons (0, 0)
- **Frame sampling**: 10 frames uniformly; object frame is one of these 10
- **Image preprocessing**: Resize to 224×224, ImageNet normalization

## Code Style

- PEP 8 compliant
- Type hints throughout
- Docstrings for all modules, classes, and functions
- References to paper equations/sections in comments

## References

Original paper: "Pose-Based Two-Stream Relational Networks for Action Recognition in Videos"  
arXiv:1805.08484

## License

This implementation follows the paper's methodology. Please cite the original paper if you use this code.

