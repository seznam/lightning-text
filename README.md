# LightningText

Lightning-fast text classification with scikit-learn integration.

LightningText is an adapter for using
[FastText's Python module](https://fasttext.cc/docs/en/python-module.html)
with
[scikit-learn](https://scikit-learn.org/), enabling easy use of scikit-learn's
features (cross validation, various metrics, multi-output, ...) with FastText.

Please note that while this project strives for maximum possible compatibility
with scikit-learn, it is not currently possible to pass all tests executed by
[`check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html),
mostly due to FastText's behavior.

While this project builds upon both FastText and scikit-learn, it is an
independent project not associated with either of the two.

## Table of Contents

- [Installation](#installation)
- [API overview](#api-overview)
  - [Base FastText API wrappers](#base-fasttext-api-wrappers)
  - [Scikit-learn compatible FastText classifier](#scikit-learn-compatible-fasttext-classifier)
    - [Note on labels representation](#note-on-labels-representation)
  - [Text preprocessing](#text-preprocessing)
  - [FastText dataset API](#fasttext-dataset-api)
  - [Hyperparameter search using Optuna](#hyperparameter-search-using-optuna)
  - [Additional scoring utilities](#additional-scoring-utilities)
- [Examples](#examples)
  - [Training and evaluating a model on train-test split](#training-and-evaluating-a-model-on-train-test-split)
  - [Multi-label classification](#multi-label-classification)
  - [K-fold Cross-validation](#k-fold-cross-validation)
  - [Hyperparameter search example with Optuna](#hyperparameter-search-example-with-optuna)
- [License](#license)

## Installation

```sh
pip install lightning-text
```

## API overview

### Base FastText API wrappers

These are thin wrappers of the APIs exposed by FastText's Python module,
providing just explicit declaration of parameters, their default values and
documentation. They pass the arguments directly to their respective counterparts
in FastText module:

- `tokenize`
- `load_model`
- `train_supervised`
- `train_unsupervised`

### Scikit-learn compatible FastText classifier

The `FastTextClassifier` class wraps supervised learning of FastText as a
scikit-learn classifier.

To ensure compatibility with scikit-learn, the classifier requires the targets
(labels) to be integers instead of strings. One way to encode string labels to
integers is to use scikit-learn's
[`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
or
[`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html),
a more convenient way for handling a dataset in FastText format is to use the
`Dataset` class (see [below](#fasttext-dataset-api)).

The classifier also adds support for pickling, including serialization of the
underlying FastText model.

#### Note on labels representation

Scikit-learn uses integers to represent classification targets (classes), and
by default, these are used as label names when fitting the underlying FastText
model.

If, however, text representation (usually original names) of the classes are
desired to be known by the FastText model, (e.g. if deploying the final model in
a stand-alone way), a label encoder can be passed to `FastTextClassifier`'s
`fit()` method using the `label_encoder` parameter. `FastTextClassifier`
supports both
[`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
(for binary and multi-class classification) and
[`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)
(for multi-label classification).

Note that if you use the `label_encoder` parameter, the class names must not
contain whitespace, otherwise you'll encounter exceptions during inference.

### Text preprocessing

The module provides the `preprocess_text` utility for basic preprocessing of
raw text for FastText. The function also provides optional removal of HTTP(S)
URLs from text.

### FastText dataset API

The `Dataset` class provides a convenient way of handling existing FastText
datasets, while representing the dataset in a scikit-learn-compatible way.

Loading a FastText dataset using `Dataset.load_from` or `Dataset.from_file` will
automatically convert the string labels to integers, using either scikit-learn's
[`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
or
[`MultiLabelBinarizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)
(if any sample has multiple labels assigned) to do the conversion. These class
methods return the used label encoder with the created dataset for converting a
fitted model's predictions back to text labels using the `inverse_transform`
method on the encoder.

### Hyperparameter search using Optuna

LightningText provides a scikit-learn-style [Optuna](https://optuna.org/)-powered
hyperparameter search with cross-validation, matching the APIs of other
scikit-learn's hyperparameter searches, e.g.
[`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).
See the `OptunaSearchCV` class for details.

### Additional scoring utilities

LightningText provides additional utilities for interpreting scores:

- `get_fold_scores_per_candidate` - Takes results of hyperparameter search with
  cross-validation (the `cv_results_` field) and returns a dictionary of metric
  names to 2D numpy array of shape `(n_candidates, n_folds)`.
- `robust_cv_score` - Returns a harmonic mean of the provided scores. Harmonic
  mean puts more weight on lower scores, naturally penalizing high variance.
- `penalized_cv_score` - Explicitly penalize the scores by their standard
  deviation and the provided `penalty_weight`.
- `stability_score` - Measure how many scores are within `threshold` of mean and
  return the ratio.

## Examples

### Training and evaluating a model on train-test split

This example demonstrates use with a binary or multi-class (single-label)
classification problem.

```python
from lightning_text import FastTextClassifier
from sklearn.metrics import classification_report

classifier = FastTextClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Multi-label classification

There are two options for multi-label classification:

#### Using the `ova` (one-vs-all) loss

```python
from lightning_text import FastTextClassifier
from sklearn.metrics import classification_report, hamming_loss
from sklearn.multioutput import MultiOutputClassifier

classifier = FastTextClassifier(
    loss='ova,
)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(f'Hamming loss: {hamming_loss(Y_test, Y_pred)}')
print(classification_report(Y_test, Y_pred))
```

The classifier will be a faster to fit and a occupy smaller space when saved,
however, requires tuning the decision threshold for its `predict()` method to be
useful (see
[`TunedThresholdClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)
as one option for achieving this).

### Training a binary classifier for each class using scikit-learn's `MultiOutputClassifier` meta-estimator

```python
from lightning_text import FastTextClassifier
from sklearn.metrics import classification_report, hamming_loss
from sklearn.multioutput import MultiOutputClassifier

# Binarizes the problem and trains a FastTextClassifier for predicting the label
# for each individual class.
classifier = MultiOutputClassifier(
    FastTextClassifier(
        verbose=0,
    ),
    n_jobs=4,
)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(f'Hamming loss: {hamming_loss(Y_test, Y_pred)}')
print(classification_report(Y_test, Y_pred))
```

This will train a binary FastText classifier for detecting each class using the
one-vs-all strategy and the resulting classifier will be usable right after
fitting, however the classifier will be slower to fit, predict and will occupy
larger space when saved, which could make it impractical if the number of
classes is large.

### K-fold Cross-validation

Note that the use of
[`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
is likely beneficial for correct evaluation a binary classification problem.
Stratification is hard to impractical or impossible for multi-label problems.

```python
from lightning_text import FastTextClassifier
from sklearn.model_selection import cross_validate

classifier = FastTextClassifier()
cv = cross_validate(
    classifier,
    X,
    y,
    cv=5,
    scoring='f1',
    n_jobs=4,
)

print(cv)
```

### Hyperparameter search example with Optuna

```python
from typing import Any

from lightning_text import FastTextClassifier
from lightning_text.optuna import (
    OptunaSearchCV,
    DEFAULT_SUPERVISED_TRAINING_HYPERPARAMETERS_SPACE,
)
import optuna
from sklearn.metrics import fbeta_score, make_scorer


def metrics_to_optuna_goals(metrics: dict[str, Any]) -> float:
    last_mean_fbeta = metrics['mean_test_score'][-1]
    return last_mean_fbeta


tries = 128

estimator = FastTextClassifier(
    verbose=0,
)

study = optuna.create_study(direction='maximize')
search = OptunaSearchCV(
    estimator=estimator,
    study=study,
    hyperparameters_space=DEFAULT_SUPERVISED_TRAINING_HYPERPARAMETERS_SPACE,
    n_iter=tries,
    scoring=make_scorer(fbeta_score, pos_label=1, beta=1),
    optuna_metrics_exporter=metrics_to_optuna_goals,
    n_jobs=4,
    refit='fbeta',
    cv=5,
    show_progress_bar=True,
)
search.fit(X, y)

print(search.cv_results_)
print(search.best_params_)
print(search.best_score_)

best_estimator = search.best_estimator_
```

## License

`lightning_text` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
