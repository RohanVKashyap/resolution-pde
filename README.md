# Resolution Generalization in Neural PDE Solvers

This repository investigates **resolution generalization** in neural PDE solvers — the ability of neural operators trained at one spatial resolution to generalize to unseen resolutions. The central research question is: *how does the training resolution distribution affect a model's ability to generalize across resolutions?*

## PDE Benchmarks

| PDE | Dimensions | Data Format | Native Resolution |
|---|---|---|---|
| Kuramoto-Sivashinsky (KS) | 1D spatial + time | HDF5 | 512 |
| Burgers' equation | 1D spatial + time | HDF5 (PDEBench) | 1024 |
| Navier-Stokes (vorticity) | 2D spatial + time | HDF5 / `.mat` | 256 |

## Model Architectures

| Model | Description |
|---|---|
| **FNO** (1D/2D) | Standard Fourier Neural Operator |
| **FFNO** (1D/2D) | Factorized FNO — applies 1D FFT per axis separately, reducing complexity from O(n²) to O(n) |
| **CNO** (1D/2D) | Convolutional Neural Operator — U-Net-like encoder-decoder with antialiased activations |
| **UNet** (1D/2D) | Classic U-Net with skip connections |
| **S4** | State Space Models (S4/S4D) adapted for PDE solving |
| **Poseidon** | Pretrained transformer (`camlab-ethz/Poseidon-B`) via HuggingFace |

## Training Strategies

Three resolution-handling strategies are compared:

1. **Naive single-resolution** — Train at one fixed grid size, test at others
2. **Resize-based** — Bicubically interpolate all data to a fixed training resolution
3. **True multi-resolution** — Mix samples at multiple native resolutions per epoch using `ResolutionGroupedDataLoader`

## Project Structure

```
├── main_1d.py                  # Entry point for 1D PDEs (KS, Burgers)
├── main_2d.py                  # Entry point for 2D PDEs (Navier-Stokes)
├── autoregressive_eval.py      # Standalone autoregressive rollout evaluation
├── frequency_evaluation.py     # Frequency-domain error analysis across checkpoints
├── run1.sh                     # SLURM batch job script
├── conf/                       # Hydra configuration hierarchy
│   ├── config.yaml
│   ├── model/                  # Per-model configs
│   ├── dataset/                # Per-dataset configs (burger/, ks/, ns/)
│   └── training/
├── models/                     # Neural operator architectures
│   ├── fno.py                  # FNO1d / FNO2d
│   ├── ffno.py                 # FFNO1D / FFNO2D
│   ├── spectral_convolution.py # Spectral & factorized spectral conv layers
│   ├── CNO1d.py / CNO2d.py     # Convolutional Neural Operator
│   ├── unet.py                 # UNet1d / UNet2d
│   └── s4*.py                  # State Space Model variants
├── dataloaders/                # Dataset loading per PDE + strategy
├── train/                      # Training loops & multi-res dataloading
│   ├── training.py             # Standard train/evaluate
│   ├── mres_training.py        # ResolutionGroupedDataLoader
│   └── interpolate_training.py # CNO resize-based training
├── utils/                      # Loss, frequency analysis, plotting, normalization
│   ├── loss.py                 # Relative L2 loss
│   ├── autoregressive_step.py  # Rollout evaluation
│   ├── low_pass_filter.py      # FFT-based low-pass filtering
│   └── multiresolution_analysis.py
└── data_generation/            # Navier-Stokes pseudo-spectral solver
```

## Setup

### Dependencies

```
torch
torchvision
numpy
scipy
h5py
einops
matplotlib
tqdm
wandb
hydra-core
omegaconf
pandas
```

Optional: `scOT` (for Poseidon transformer)

### Install

```bash
conda create -n pde python=3.10
conda activate pde
pip install torch torchvision numpy scipy h5py einops matplotlib tqdm wandb hydra-core omegaconf pandas
```

## Usage

Configuration is managed via [Hydra](https://hydra.cc/). Override model and dataset configs from the command line.

### Training

**Single-resolution (1D):**
```bash
python main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive
python main_1d.py model=unet/unet_1d dataset=burger/burger_naive
```

**True multi-resolution (1D):**
```bash
python main_1d.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
python main_1d.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1
```

**2D Navier-Stokes:**
```bash
python main_2d.py model=ffno_2d/ffno_2d dataset=ns/ns_naive_true_mres1
python main_2d.py model=cno_2d/cno_2d_original dataset=ns/ns_naive_true_mres1
```

### Evaluation

**Frequency-domain analysis:**
```bash
python frequency_evaluation.py model=ffno_1d/ffno_1d dataset=burger/burger_naive_true_mres1
```

**Autoregressive rollout:**
```bash
python autoregressive_eval.py model=ffno_1d/ffno_1d dataset=ks/ks_naive_true_mres1
```

### SLURM

```bash
sbatch run1.sh
```

## Evaluation Pipeline

After training, models are evaluated on:

1. **Super-resolution generalization** — tested at every resolution in `[32, 64, 128, ..., max]`
2. **Autoregressive rollout** — multi-step predictions where each output is fed back as the next input
3. **Frequency decomposition** — prediction error broken down by Fourier mode to identify which frequency components degrade under resolution mismatch

Results are logged to [Weights & Biases](https://wandb.ai/) and saved as checkpoints under `checkpoints/`.

## Loss Function

Relative L2 loss: `||pred - target||₂ / ||target||₂`, averaged over the batch.
