📦 cytocoarsening.py
=======================

We want to identify cell-types that are enriched for both phenotype (e.g. cell phenotype) and relate to the external information. Graph-based approaches for identifying these modules can suffer in the single-cell setting because there is an extremely large number of cells profiled per sample and we often profile multiple samples with multiple different experimental conditions or timepoints.

Overview
=======================
<p align="center">
  <img src="/doc/intuitive_coarsening_illustration.jpg"/>
  <img src="/doc/Cytocoarsening.png"/>
</p>

Installation
-----

```bash
pip install cytocoarsening
```

* Or you can clone the git repository by, 

```
git clone https://github.com/ChenCookie/cytocoarsening.git
```

* Once you've clone the repository, please change your working directory into this folder.

```
cd cytocoarsening
```

Dataset
--------------

-   [What is setup.py?] on Stack Overflow
-   [Official Python Packaging User Guide](https://packaging.python.org)
-   [The Hitchhiker's Guide to Packaging]
-   [Cookiecutter template for a Python package]

Parameter Explanation
--------------
The function can be excute at one line
```
coarsening_group,group_edge,result_dicts=cytocoarsening(cell_data,cell_label,multipass,k_nearest_neighbors)
```
input
* `cell_data` - unintegrated spliced gene expression data
* `cell_label` - unintegrated spliced gene expression data
* `multipass` - unintegrated spliced gene expression data
* `k_nearest_neighbors` - unintegrated spliced gene expression data

output
* `coarsening_group` - unintegrated spliced gene expression data
* `group_edge` - unintegrated spliced gene expression data
* `result_dicts` - unintegrated spliced gene expression data

Toy Example
--------------

```
from cytocoarsening.cytocoarsening import cytocoarsening
import numpy as np
import random

cell_data=[[random.random() for i in range(33)] for j in range(4500)]
cell_data=np.array(cell_data)

cell_label = np.array([0] * 1000 + [1] * (3500))
np.random.shuffle(cell_label)

group,edge,diccts=cytocoarsening(cell_data,cell_label,3,5)
```