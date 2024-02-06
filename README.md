<br/>
<p align="center"><img src="img/tunetables_logo.png" width=700 /></p>

----
![Crates.io](https://img.shields.io/crates/l/Ap?color=orange)

We introduce TuneTables, a tabular classification algorithm that overcomes the limitations of prior-data fitted networks to achieve strong performance on large datasets.


While TabPFN achieves very strong performance on small tabular datasets, its current limitations include fewer than 1000 datapoints, fewer than 100 features, and fewer than 10 class labels. In this work, we overcome these limitations and substantially improve the performance of PFNs by developing context optimization techniques; specifically, we propose TuneTables, a novel prompt-tuning strategy. TuneTables scales TabPFN to be competitive with state-of-the-art tabular classification methods on larger datasets, while having additional benefits as well: (1) substantially lower inference time than TabPFN, (2) can be used as an interpretability tool, and (3) can mitigate biases by optimizing a fairness objective.

<p align="center"><img src="img/tunetables_overview.png" width=700 /></p>

This codebase extends the excellent public repository [TabPFN]([xxx](https://github.com/automl/tabpfn)), by Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter.

## Installation

```bash
pip install tabpfn
```

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))
```

### TabPFN Usage

TabPFN is different from other methods you might know for tabular classification.
Here, we list some tips and tricks that might help you understand how to use it best.

- Do not preprocess inputs to TabPFN. TabPFN pre-processes inputs internally. It applies a z-score normalization (`x-train_x.mean()/train_x.std()`) per feature (fitted on the training set) and log-scales outliers [heuristically](https://github.com/automl/TabPFN/blob/f7402ec1916aa78d953574daf95508045af5953e/tabpfn/utils.py#L201). Finally, TabPFN  applies a PowerTransform to all features for every second ensemble member. Pre-processing is important for the TabPFN to make sure that the real-world dataset lies in the distribution of the synthetic datasets seen during training. So to get the best results, do not apply a PowerTransformation to the inputs.
- TabPFN expects scalar values only (you need to encode categoricals as integers e.g. with [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder)). It works best on data that does not contain any categorical or NaN data (see [Appendix B.1](https://arxiv.org/abs/2207.01848)).
- TabPFN ensembles multiple input encodings per default. It feeds different index rotations of the features and labels to the model per ensemble member. You can control the ensembling with `TabPFNClassifier(...,N_ensemble_configurations=?)`
- TabPFN does not use any statistics from the test set. That means predicting each test example one-by-one will yield the same result as feeding the whole test set together.
- TabPFN is differentiable in principle, only the pre-processing is not and relies on numpy.


## TabPFN-PT notes

* Everything expects to run from TabPFN-pt/tabpfn
