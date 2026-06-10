# MetaSort

This repository contains the current MetaSort algorithm code, packaged example tissue data,
and notebooks for tissue deconvolution experiments.

## Contents

- `metasort/algorithm.py`: base weighted deconvolution utilities.
- `metasort/metasort.py`: MetaSort solver with Hessian, average-gradient, residual, and regularization losses.
- `data/`: packaged example tissues (`Blood`, `Eye`, `Fat`, `Lung`).
- `notebook_test/tissue_demo.ipynb`: runs packaged tissue mixtures and reports accuracy.
- `notebooks/all_tissues_method_comparison.ipynb`: compares MetaSort against external method outputs.
- `notebooks/synthetic_subtype_demo.ipynb`: synthetic subtype experiment.
- `notebooks/reference_recovery_mean_spearman_demo.ipynb`: reference recovery experiment.

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
jupyter notebook notebook_test/tissue_demo.ipynb
```

The notebook uses the packaged example data under `data/`.

## Minimal Python Usage

```python
from pathlib import Path

from metasort import MetaSortSolver, load_bulk_signature_inputs

data_root = Path("data/Blood")
signature, bulk, genes, cell_types = load_bulk_signature_inputs(
    data_root,
    mixture_name="Mixture1",
)

solver = MetaSortSolver()
result = solver.solve(signature, bulk, cell_types=cell_types)

print(result.cell_types)
print(result.proportions)
```

## Hierarchical MetaSort Usage

`HierarchicalMetaSortSolver` can build the cell-type similarity tree from the
single-cell reference (`singleCellExpr.txt`, `singleCellLabels.txt`, and
optionally `singleCellSubjects.txt`). It then cuts the tree into coarse groups,
estimates those group proportions with MetaSort, expands the tree one level at a
time, performs split-specific gene selection for each expanded parent node,
estimates child proportions with parent-local MetaSort solves, and constrains
child node proportions by the parent proportions estimated in the previous stage.

```python
from pathlib import Path

from metasort import (
    HierarchicalMetaSortSolver,
    load_bulk_signature_inputs,
    load_single_cell_hierarchy_inputs,
)

data_root = Path("data/Blood")
signature, bulk, genes, cell_types = load_bulk_signature_inputs(
    data_root,
    mixture_name="Mixture1",
)
single_cell_expr, single_cell_labels, single_cell_subjects = load_single_cell_hierarchy_inputs(data_root)

solver = HierarchicalMetaSortSolver()
result = solver.solve(
    signature,
    bulk,
    cell_types=cell_types,
    single_cell_expr=single_cell_expr,
    single_cell_labels=single_cell_labels,
    single_cell_subjects=single_cell_subjects,
)

print(result.cell_types)
print(result.proportions)
print(result.hierarchy_source)
print(result.hierarchy)
```
