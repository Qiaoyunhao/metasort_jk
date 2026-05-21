# MetaSort

This repository contains the current MetaSort algorithm code and a tissue demo notebook.

## Contents

- `metasort/algorithm.py`: base weighted deconvolution utilities.
- `metasort/metasort.py`: MetaSort solver with Hessian, average-gradient, residual, and regularization losses.
- `data/`: packaged example tissues (`Fat`, `Blood`, `Lung`).
- `notebooks/tissue_demo.ipynb`: runs `Fat`, `Blood`, and `Lung` `Mixture1` to `Mixture20` and reports accuracy.

## Demo Configuration

The packaged default configuration matches the current tested parameter version:

```text
lambda_hessian = 1.0
lambda_avg_gradient = 1.0
lambda_residual = 1.5
lambda_gene_importance = 0.0
lambda3 = 0.01
lambda4 = 0.001
convergence_tol = 0.005
final_weight_max = 10.0
```

Accuracy in the notebook is reported as:

```text
accuracy = 1 - L1 / 2
```

## Install

From the repository root:

```bash
pip install -r requirements.txt
```

## Run Demo

From the repository root:

```bash
jupyter notebook notebooks/tissue_demo.ipynb
```

The notebook uses the packaged example data under `data/`.
