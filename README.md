# COMP2000 Assessment 1 - Code

Notebooks: Exercise1_MLP, Exercise2_CNN, Exercise3_Regularisation, Exercise4_Width_Depth (each is self-contained).

## Reproduce
1. `pip install -r requirements.txt`
2. `jupyter notebook`, open a notebook and choose *Run All*. CIFAR-10 downloads to `./data` automatically.
3. Seeds are fixed (train/val split seed 42; run seeds 0/1/2), so results are reproducible up to GPU non-determinism.

## Settings shared by all exercises
Adam, lr 1e-3, batch 128, 30 epochs, cross-entropy, 45k/5k train/val split of the official training set; the official test set is used only for final reporting. The best-validation-accuracy epoch is restored before testing.

## Outputs
CSV tables (`ex*_results/summary.csv`) and figures (`./figures/*.png`) used in the report.
Approx. runtime on a single GPU: Ex1 ~3 min, Ex2 ~5 min, Ex3 ~40 min, Ex4 ~15 min (much longer on CPU; reduce EPOCHS if needed and state it in the report).
