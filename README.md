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
to run the experiments from the original
[paper](https://doi.org/10.1007/978-3-031-08011-1_7).

The remaining files in this repo are scripts for performing various experiments
using S-OCT, our reimplementations of OCT/OCT-H and FlowOCT, optimal
classification trees from [Interpretable AI](https://www.interpretable.ai/),
and [PyDL8.5](https://www.ijcai.org/Proceedings/2020/0750.pdf).

For full details of the S-OCT model, please see the original
[paper](https://doi.org/10.1007/978-3-031-08011-1_7), Shattering Inequalities
for Learning Optimal Decision Trees.
