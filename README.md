# S-OCT
S-OCT is a mixed-integer programming (MIP) formulation for training an optimal
multivariate decision tree. For ease of use, the model is implemented as a
[scikit-learn](https://scikit-learn.org/stable/) classifier, thus it can be
used with model selection and evaluation tools such as `pipeline.Pipeline`,
`model_selection.GridSearchCV`, and `model_selection.cross_validate`, as
demonstrated in the script `soct_comprehensive.py`.

Our code for S-OCT, as well as our reimplementations of
[OCT/OCT-H](https://www.mit.edu/~dbertsim/papers/Machine%20Learning%20under%20a%20Modern%20Optimization%20Lens/Optimal_classification_trees_MachineLearning.pdf)
and [FlowOCT](https://arxiv.org/abs/2103.15965), can be found in the `src`
folder.

The `datasets` folder contains datasets from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
The file `datasets.py` contains code for loading these datasets, as well as
scikit-learn compatible transformers for performing bucketization, a
preprocessing step needed by algorithms that assume binary features.

The `CPAIOR2022` folder is here for archival purposes. It contains codes needed
to run the experiments from our
[CPAIOR 2022 paper](https://doi.org/10.1007/978-3-031-08011-1_7).

The remaining code files in this repo are scripts to perform experiments in our
[Constraints paper](https://doi.org/10.1007/s10601-023-09367-y).
`parameter_tuning.py` runs the tuning experiments described in Section 6.1.1.
The `*_mip.py` scripts run the direct MIP comparison described in Section 6.1.2;
these scripts write results to `mip_comparison.csv`. The `*_comprehensive.py`
scripts run the comprehensive comparison desribed in Section 6.1.3; these
scripts write results to `comprehensive.csv`. In addition to the models in
`src`, our experiments also test models from
[Interpretable AI](https://www.interpretable.ai/) and
[PyDL8.5](https://www.ijcai.org/Proceedings/2020/0750.pdf).

### References
1. Boutilier, J., Michini, C., Zhou, Z. (2022). *Shattering Inequalities for Learning Optimal Decision Trees.* Proceedings of CPAIOR 2022. [DOI:10.1007/978-3-031-08011-1_7](https://doi.org/10.1007/978-3-031-08011-1_7) (**Best paper award**)
2. Boutilier, J., Michini, C., Zhou, Z. (2023). *Optimal multivariate decision trees.* Constraints. [DOI:10.1007/s10601-023-09367-y](https://doi.org/10.1007/s10601-023-09367-y)
