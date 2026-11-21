# PUSL3123 AI and Machine Learning – Assessment 1: Set Exercises (Group 18)

Neural-network image classification on **CIFAR-10** with PyTorch. Four notebooks, one per exercise, investigate
MLP vs CNN architectures, regularisation, and the effect of model width and depth.

| Notebook | Exercise | What it does | Approx. runtime (Tesla T4 GPU) |
| --- | --- | --- | --- |
| `Exercise_1_MLP.ipynb` | 1 – MLP classifier | Three-layer MLP (3072→512→256→10), loss/accuracy curves, confusion matrix | ~6 min |
| `Exercise2_CNN.ipynb` | 2 – CNN vs MLP | Three-layer CNN vs retrained MLP under identical settings; parameter counts, confusion matrices, learnt filters | ~12 min |
| `Exercise3_PUSL3123.ipynb` | 3 – Regularisation | Baseline CNN vs dropout, data augmentation and both combined; 3 seeds each (12 runs) | ~1.5 h |
| `Exercise4_Width_Depth.ipynb` | 4 – Width and depth | Three widths and three depths; 3 seeds each (18 runs) | ~1.75 h |

Each notebook is self-contained and can be run on its own.

## 1. Requirements

- Python 3.10 or newer (developed on Python 3.13, PyTorch 2.14, torchvision 0.29, CUDA 13.0)
- A CUDA GPU is strongly recommended. The code falls back to Apple MPS or CPU automatically, but Exercises 3–4 take many hours on a CPU.
- About 200 MB of free disk space for the dataset.

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU support, install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/>.

## 2. How to run

### Option A – Google Colab (recommended)
1. Open a notebook in Colab (File → Upload notebook, or open from the GitHub repository).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run **Runtime → Run all**.

### Option B – Locally
```bash
pip install -r requirements.txt
jupyter notebook
```
Open a notebook and choose **Kernel → Restart & Run All**.

CIFAR-10 is downloaded automatically to `./data` on the first run through `torchvision.datasets.CIFAR10`.

## 3. Experimental protocol (identical in every notebook)

| Setting | Value |
| --- | --- |
| Dataset split | 50,000 training images → 45,000 train / 5,000 validation (fixed seed 42); 10,000 test images kept separate |
| Preprocessing | `ToTensor()` + per-channel normalisation (mean 0.4914, 0.4822, 0.4465; std 0.2470, 0.2435, 0.2616) |
| Loss / optimiser | Cross-entropy / Adam, learning rate 0.001 |
| Batch size / epochs | 128 / 30 |
| Model selection | Weights from the epoch with the highest **validation** accuracy |
| Test set | Evaluated **once**, on the selected weights only. Never used for training or tuning. |
| Seeds | Exercises 1–2: one seed. Exercises 3–4: seeds 0, 1, 2 (mean ± SD reported) |

Exercise 1 uses `random_split(..., generator=manual_seed(42))`. Exercises 2–4 use `torch.randperm(50000, generator=manual_seed(42))`.
Both produce **exactly the same** train/validation indices, so every exercise uses the same partitions.

## 4. Reproducibility notes

- Seeds are set for Python, NumPy and PyTorch (`set_seed`) and for each `DataLoader` generator.
- GPU convolution kernels are not fully deterministic, so re-runs can differ slightly (typically < 0.5 percentage points). The 3-seed mean ± SD in Exercises 3–4 accounts for this.
- **Resumable runs:** Exercises 3 and 4 save progress to `ex3_checkpoint.json` / `ex4_checkpoint.json` after every training run. If a Colab session disconnects, re-run the notebook and finished runs are skipped. **Delete the checkpoint file to start from scratch.**

## 5. Outputs

| Notebook | Files produced |
| --- | --- |
| Exercise 1 | `ex1_results.csv`, `mlp_baseline.pth`, `mlp_loss_curve.png`, `mlp_accuracy_curve.png`, `mlp_confusion_matrix.png` |
| Exercise 2 | `ex2_results.csv`, `figures/ex2_loss_mlp_vs_cnn.png`, `figures/ex2_cnn_curves.png`, `figures/ex2_conf_mlp.png`, `figures/ex2_conf_cnn.png`, `figures/ex2_filters.png` |
| Exercise 3 | `ex3_raw_results.csv` (per seed), `ex3_summary.csv` (mean ± SD), `figures/ex3_comparison.png`, `figures/ex3_loss_curves.png` |
| Exercise 4 | `ex4_raw_results.csv`, `ex4_summary.csv`, `ex4_summary_formatted.csv`, `figures/ex4_summary.png`, `figures/ex4_loss_curves.png` |

## 6. Key results (test accuracy)

| Model | Test accuracy | Trainable parameters |
| --- | --- | --- |
| MLP (Ex. 1) | 52.42% | 1,707,274 |
| CNN [32, 64] (Ex. 2) | 71.90% | 60,362 |
| CNN + data augmentation (Ex. 3) | 76.60 ± 0.40% | 60,362 |
| CNN, 5 conv layers (Ex. 4, D3) | 73.32 ± 0.48% | 43,018 |

The full analysis is in the group report (PDF).

## 7. Repository structure

```
.
├── Exercise_1_MLP.ipynb
├── Exercise2_CNN.ipynb
├── Exercise3_PUSL3123.ipynb
├── Exercise4_Width_Depth.ipynb
├── requirements.txt
├── README.md
```

## 8. Authors – Group 18

| Member | Exercise |
| --- | --- |
| [Name 1] ([ID]) | Exercise 1 |
| [Name 2] ([ID]) | Exercise 2 |
| [Name 3] ([ID]) | Exercise 3 |
| [Name 4] ([ID]) | Exercise 4 |
