# MetaSort

This repository contains the current MetaSort algorithm code, packaged example tissue data,
and notebooks for tissue deconvolution experiments.

## Contents

- `metasort/algorithm.py`: preprocessing and simplex-constrained least-squares utilities.
- `metasort/metasort.py`: MetaSort solver with Hessian, average-gradient, residual, and regularization losses.
- `metasort/anchor_recovery.py`: anchor-guided NNLS deconvolution and non-anchor reference recovery.
- `data/`: packaged example tissues (`Blood`, `Eye`, `Fat`, `Lung`).
- `notebook_test/tissue_demo.ipynb`: runs packaged tissue mixtures and reports accuracy.
- `notebooks/all_tissues_method_comparison.ipynb`: compares MetaSort against external method outputs.
- `notebooks/synthetic_subtype_demo.ipynb`: synthetic subtype experiment.
- `notebooks/reference_recovery_mean_spearman_demo.ipynb`: reference recovery experiment.

## Demo Configuration

The packaged default configuration matches the current tested parameter version:

```text
lambda_hessian = 0.03
lambda_avg_gradient = 0.0
lambda_residual = 0.02
lambda_gene_importance = 0.0
lambda3 = 0.0001
lambda4 = 1e-05
convergence_tol = 0.005
averaging_old_weight = 2.5
use_sqrt_sphere_hessian = True
meta_weight_floor = 1.0
meta_weight_baseline = 10.0
normalize_meta_weight_mean = True
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

When inputs are loaded with `load_bulk_signature_inputs`, each reference
signature column and the selected bulk mixture vector are first normalized to
sum to 1 over the shared genes. The reference columns and selected bulk vector
are then concatenated and z-scored gene by gene with the same row mean and
standard deviation.

After this preprocessing, initial and weighted proportion solves use
simplex-constrained least squares (`p >= 0`, `sum(p) = 1`) rather than NNLS
followed by post-hoc normalization.

By default the MetaSort Hessian spectrum is computed in the standard simplex
tangent space. Setting `use_sqrt_sphere_hessian=True` maps the current
proportions with `u = sqrt(p)` onto the unit sphere and computes the local
Gauss-Newton Hessian spectrum in the tangent space at `u`.

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

## Anchor-Guided Reference Recovery

`AnchorRecoverySolver` first runs MetaSort on all genes to score genes by
residual fit and learned MetaSort weight. Genes with small residuals and large
weights are selected as anchor genes. The final proportions are then estimated
with simplex-constrained least squares on only the anchor genes, and non-anchor
genes are recovered with a reference-regularized least-squares solve:

```text
min_S ||B_non_anchor - S_non_anchor P_anchor||^2
    + lambda_reference ||S_non_anchor - S_sc,non_anchor||^2
```

Anchor genes keep their original reference expression, so they define the
coordinate system used for deconvolution. Non-anchor genes are allowed to shift
toward the bulk domain.

By default, anchor selection and anchor-only proportion estimation use MetaSort-style
preprocessing: reference columns and bulk samples are normalized to sum to 1,
then jointly z-scored per gene before computing residuals, MetaSort weights,
and anchor proportions. Anchor filtering uses the residual RMS in this
preprocessed space by default; setting `anchor_residual_metric="relative"`
switches back to residual RMS divided by the fitted/bulk RMS scale. Anchor
proportions are solved with `p >= 0` and
`sum(p) = 1`; setting `anchor_proportion_method="nnls"` switches back to NNLS
followed by normalization. Reference recovery still uses the original input
expression scale so recovered signatures remain interpretable.

```python
from pathlib import Path

from metasort import AnchorRecoverySolver, load_bulk_signature_inputs

data_root = Path("data/Blood")
signature, bulk, genes, cell_types = load_bulk_signature_inputs(
    data_root,
    mixture_name="Mixture1",
)

solver = AnchorRecoverySolver()
result = solver.solve(
    signature,
    bulk,
    cell_types=cell_types,
    genes=genes,
)

print(result.proportions)
print(result.anchor_genes[:10])
print(result.reconstruction_error)
```

## Hierarchical MetaSort Usage

`HierarchicalMetaSortSolver` uses a manually supplied cell-type tree. It cuts
the tree into coarse groups, estimates those group proportions with MetaSort,
expands the tree one level at a time, performs split-specific gene selection for
each expanded parent node, estimates child proportions with parent-local
MetaSort solves, and constrains child node proportions by the parent proportions
estimated in the previous stage.

The batch comparison script runs plain MetaSort by default. Supplying
`--hierarchy-file path/to/hierarchy.json` switches it to hierarchical
deconvolution; omitting that argument keeps direct deconvolution.

Manual hierarchy JSON uses the same node shape as `HierarchyNode.to_dict()`.
Internal node `cell_types` can be omitted and inferred from children. Leaf node
names must exactly match the signature cell-type column names.

```json
{
  "name": "root",
  "children": [
    {
      "name": "lymphoid",
      "children": [
        {"name": "B cell"},
        {"name": "T cell"}
      ]
    },
    {
      "name": "myeloid",
      "children": [
        {"name": "Monocyte"},
        {"name": "Neutrophil"}
      ]
    }
  ]
}
```

```python
from pathlib import Path

from metasort import (
    HierarchicalMetaSortSolver,
    HierarchyNode,
    load_bulk_signature_inputs,
)

data_root = Path("data/Blood")
signature, bulk, genes, cell_types = load_bulk_signature_inputs(
    data_root,
    mixture_name="Mixture1",
)
hierarchy = HierarchyNode(
    name="root",
    cell_types=cell_types,
    children=[
        HierarchyNode(name=cell_type, cell_types=[cell_type])
        for cell_type in cell_types
    ],
)

solver = HierarchicalMetaSortSolver()
result = solver.solve(
    signature,
    bulk,
    cell_types=cell_types,
    hierarchy=hierarchy,
)

print(result.cell_types)
print(result.proportions)
print(result.hierarchy_source)
print(result.hierarchy)
```
