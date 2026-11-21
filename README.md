# PUSL3123 AI and Machine Learning – Assessment 1: Set Exercises (Group 18)

Neural-network image classification on **CIFAR-10** with PyTorch. Four Jupyter notebooks, one per exercise, investigate
MLP vs CNN architectures, regularisation, and the effect of model width and depth.

| Exercise | Notebook | What it does | Runtime (Tesla T4 GPU) | Open in Colab |
| --- | --- | --- | --- | --- |
| 1 – MLP classifier | [`Exercise_1_MLP.ipynb`](Group_18_Set_Exercises/Exercise_1/Exercise_1_MLP.ipynb) | Three-layer MLP (3072→512→256→10); loss/accuracy curves; confusion matrix | ~6 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MenadiDev/PUSL3123_Group_18/blob/main/Group_18_Set_Exercises/Exercise_1/Exercise_1_MLP.ipynb) |
| 2 – CNN vs MLP | [`Exercise2_CNN.ipynb`](Group_18_Set_Exercises/Exercise_2/Exercise2_CNN.ipynb) | Three-layer CNN vs retrained MLP under identical settings; parameter counts, confusion matrices, learnt filters | ~12 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MenadiDev/PUSL3123_Group_18/blob/main/Group_18_Set_Exercises/Exercise_2/Exercise2_CNN.ipynb) |
| 3 – Regularisation | [`Exercise3_Regularisation.ipynb`](Group_18_Set_Exercises/Exercise_3/Exercise3_Regularisation.ipynb) | Baseline CNN vs dropout, data augmentation and both combined (3 seeds each, 12 runs); Welch t-tests; 60-epoch follow-up | ~2 h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MenadiDev/PUSL3123_Group_18/blob/main/Group_18_Set_Exercises/Exercise_3/Exercise3_Regularisation.ipynb) |
| 4 – Width and depth | [`Exercise4_Width_Depth.ipynb`](Group_18_Set_Exercises/Exercise_4/Exercise4_Width_Depth.ipynb) | Three widths and three depths of CNN (3 seeds each, 18 runs) | ~1.75 h | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MenadiDev/PUSL3123_Group_18/blob/main/Group_18_Set_Exercises/Exercise_4/Exercise4_Width_Depth.ipynb) |

Each notebook is self-contained and can be run on its own. All four notebooks are submitted with their outputs, so results and figures can be read without re-running.

## 1. Repository structure

```
PUSL3123_Group_18/
├── README.md
├── requirements_Assessment1.txt
└── Group_18_Set_Exercises/
    ├── Exercise_1/
    │   ├── Exercise_1_MLP.ipynb
    │   ├── ex1_results.csv
    │   ├── mlp_loss_curve.png
    │   ├── mlp_accuracy_curve.png
    │   ├── mlp_confusion_matrix.png
    │   └── mlp_baseline.pth              # trained MLP weights
    ├── Exercise_2/
    │   ├── Exercise2_CNN.ipynb
    │   ├── ex2_results.csv
    │   ├── ex2_loss_mlp_vs_cnn.png
    │   ├── ex2_cnn_curves.png
    │   ├── ex2_conf_mlp.png
    │   ├── ex2_conf_cnn.png
    │   └── ex2_filters.png
    ├── Exercise_3/
    │   ├── Exercise3_Regularisation.ipynb
    │   ├── ex3_raw_results.csv           # one row per configuration and seed
    │   ├── ex3_summary.csv               # mean ± SD per configuration
    │   ├── ex3_significance.csv          # Welch t-test results
    │   ├── ex3_extended_60ep.csv         # 60-epoch follow-up
    │   ├── ex3_checkpoint.json           # saved training progress (see section 4)
    │   ├── ex3_comparison.png
    │   ├── ex3_loss_curves.png
    │   └── ex3_extended_60ep.png
    └── Exercise_4/
        ├── Exercise4_Width_Depth.ipynb
        ├── ex4_raw_results.csv
        ├── ex4_summary.csv
        ├── ex4_summary_formatted.csv
        ├── ex4_checkpoint.json           # saved training progress (see section 4)
        ├── ex4_summary.png
        └── ex4_loss_curves.png
```

The CSV and PNG files in each folder are the outputs of the submitted runs, and are the figures and tables used in the report.

## 2. Requirements

- Python 3.10 or newer (developed on Python 3.13, PyTorch 2.14, torchvision 0.29, CUDA 13.0)
- A CUDA GPU is strongly recommended. The code falls back to Apple MPS or CPU automatically, but Exercises 3–4 take many hours on a CPU.
- About 200 MB of free disk space for the CIFAR-10 dataset.

Install the dependencies from the repository root:

```bash
pip install -r requirements_Assessment1.txt
```

For GPU support, install the PyTorch build that matches your CUDA version from <https://pytorch.org/get-started/locally/>.
Google Colab already has every dependency installed.

## 3. How to run

### Option A – Google Colab (recommended)
1. Click an **Open in Colab** badge in the table above.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run **Runtime → Run all**.

### Option B – Locally
```bash
git clone https://github.com/MenadiDev/PUSL3123_Group_18.git
cd PUSL3123_Group_18
pip install -r requirements_Assessment1.txt
jupyter notebook
```
Open a notebook inside `Group_18_Set_Exercises/Exercise_N/` and choose **Kernel → Restart & Run All**.

CIFAR-10 is downloaded automatically by `torchvision.datasets.CIFAR10` into a `data/` folder next to the notebook on the first run.
When a notebook is run, Exercises 2–4 save their figures into a `figures/` subfolder, and Exercise 1 saves them next to the notebook.

## 4. Reproducibility notes

- **Same data partitions everywhere:** the 50,000 CIFAR-10 training images are split into 45,000 train / 5,000 validation with a fixed seed (42). Exercise 1 uses `random_split(..., generator=manual_seed(42))` and Exercises 2–4 use `torch.randperm(50000, generator=manual_seed(42))`; both produce **exactly the same** indices. The 10,000-image test set is never used for training, tuning or model selection.
- **Seeds:** Python, NumPy and PyTorch seeds are set (`set_seed`), as is each `DataLoader` generator. Exercises 1–2 use one seed; Exercises 3–4 use seeds 0, 1, 2 and report mean ± standard deviation.
- **Small run-to-run differences:** GPU convolution kernels are not fully deterministic, so re-running can change results slightly (typically < 0.5 percentage points).
- **Checkpoint files:** Exercises 3 and 4 save progress to `ex3_checkpoint.json` / `ex4_checkpoint.json` after every training run, so an interrupted Colab session can be resumed.
  - If you run the notebook **locally from this repository**, the included checkpoint is found and the finished runs are **skipped**, so the notebook reproduces the submitted results, tables and figures without retraining. (In Exercise 3 the 60-epoch follow-up has no checkpoint and still retrains, about 35 minutes on a T4 GPU.)
  - **To retrain from scratch, delete the checkpoint file first.** (Opening the notebook in Colab from GitHub does not copy the checkpoint, so Colab always retrains.)

## 5. Experimental protocol (identical in every notebook)

| Setting | Value |
| --- | --- |
| Preprocessing | `ToTensor()` + per-channel normalisation (mean 0.4914, 0.4822, 0.4465; std 0.2470, 0.2435, 0.2616) |
| Loss / optimiser | Cross-entropy / Adam, learning rate 0.001 |
| Batch size / epochs | 128 / 30 (plus a 60-epoch follow-up in Exercise 3) |
| Activation | ReLU |
| Model selection | Weights from the epoch with the highest **validation** accuracy |
| Test set | Evaluated **once**, on the selected weights only |
| Training accuracy | Re-measured in evaluation mode on un-augmented training images, so dropout and augmentation do not distort the generalisation gap |

## 6. Key results (test accuracy)

| Model | Test accuracy | Trainable parameters |
| --- | --- | --- |
| MLP (Ex. 1) | 52.42% | 1,707,274 |
| CNN [32, 64] (Ex. 2) | 71.90% | 60,362 |
| CNN + data augmentation (Ex. 3) | 76.56 ± 0.12% (79.1% with 60 epochs, seed 0) | 60,362 |
| CNN, 3 conv layers (Ex. 4, D2) | 73.20 ± 0.75% | 24,522 |
| CNN, 5 conv layers (Ex. 4, D3) | 73.32 ± 0.48% | 43,018 |

The full analysis is in the group report (PDF).

## 7. Authors – Group 18

| Member | Exercise |
| --- | --- |
| [Name 1] ([ID]) | Exercise 1 – MLP classifier |
| [Name 2] ([ID]) | Exercise 2 – CNN vs MLP |
| [Name 3] ([ID]) | Exercise 3 – Regularisation |
| [Name 4] ([ID]) | Exercise 4 – Width and depth |

PUSL3123 AI and Machine Learning · Module leader: Dr Vivek Singh
