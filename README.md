# PUSL3123
AI and Machine Learning Assessment 1: Set Exercises

Code for the group coursework investigating how network architecture, regularisation, data augmentation and model size affect image classification on CIFAR-10, using PyTorch.

## Contents

| Notebook | Exercise | What it does |
|---|---|---|
| `Exercise1_MLP.ipynb` | 1 | Three-layer MLP baseline (3072-512-256-10) |
| `Exercise2_CNN.ipynb` | 2 | Three-layer CNN (2 conv + 1 linear) compared with the MLP |
| `Exercise3_Regularisation.ipynb` | 3 | Dropout and data augmentation vs an unchanged CNN baseline (3 seeds) |
| `Exercise4_Width_Depth.ipynb` | 4 | Three widths and three depths of CNN (3 seeds) |

Each notebook is self-contained: it prepares the data, defines the models, trains and evaluates, and saves its own tables and figures.

## Requirements

- Python 3.9+
- `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `jupyter`

```bash
pip install torch torchvision numpy pandas matplotlib jupyter
```

A CUDA GPU is strongly recommended. Approximate run times on a single GPU: Exercise 1 ~5 min, Exercise 2 ~10 min, Exercise 3 ~1-2 h, Exercise 4 ~1.5-2.5 h. On a CPU they take much longer.

## How to reproduce

1. Clone the repository and install the requirements.
2. Open a notebook in Jupyter (or Colab) and choose **Restart & Run All**. Run each notebook from the same folder so they share the `./data` directory.
3. CIFAR-10 is downloaded automatically to `./data` on the first run.
4. Exercises 3 and 4 save progress to `ex3_checkpoint.json` / `ex4_checkpoint.json` after every run. If a run is interrupted, re-run the training cell and it resumes. Delete the checkpoint file to start from scratch.

## Experimental protocol (identical across all exercises)

- **Data:** the official CIFAR-10 partitions. The 50,000 training images are split into 45,000 train / 5,000 validation with a fixed seed (42). The 10,000 test images are never used for training, tuning or model selection, and are evaluated once per trained model.
- **Preprocessing:** per-channel mean/standard-deviation normalisation. Augmentation (Exercise 3 only) is applied to training images only.
- **Training:** Adam, learning rate 0.001, batch size 128, 30 epochs, cross-entropy loss.
- **Model selection:** the epoch with the best validation accuracy is restored before the final evaluation.
- **Seeds:** Exercise 1 and 2 use seed 0; Exercises 3 and 4 use seeds 0, 1, 2 and report mean ± standard deviation.

## Outputs

Each notebook writes its results to the working directory:

| Exercise | Tables / data | Figures |
|---|---|---|
| 1 | `ex1_results.csv`, `mlp_baseline.pth` | `mlp_loss_curve.png`, `mlp_accuracy_curve.png`, `mlp_confusion_matrix.png` |
| 2 | `ex2_results.csv` | `figures/ex2_*.png` (loss curves, confusion matrices, learned filters) |
| 3 | `ex3_raw_results.csv`, `ex3_summary.csv` | `figures/ex3_*.png` |
| 4 | `ex4_raw_results.csv`, `ex4_summary.csv` | `figures/ex4_*.png` |

## Key results (test accuracy)

| Exercise | Model / setting | Test accuracy |
|---|---|---|
| 1 | MLP (1.71M parameters) | 52.4% |
| 2 | CNN (60k parameters) vs MLP retrained in the same notebook | 71.9% vs 52.2% |
| 3 | Baseline / Dropout / Augmentation / Both (mean of 3 seeds) | 71.3% / 75.7% / 76.6% / 75.2% |
| 4 | Width and depth variants | see `ex4_summary.csv` |

Small differences (a few tenths of a percent) between notebooks for the same architecture and seed are expected from GPU non-determinism and hardware differences.

## Notes

- The numbers above come from the submitted runs; the report and the CSV files in this repository are consistent with them.
- If a notebook hangs on Windows at the first training step, set `NUM_WORKERS = 0` in the setup cell. This does not change the results.
- Training and evaluation use only the standard `torchvision.datasets.CIFAR10` data.

## Team

| Name | Student ID | Contribution |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
