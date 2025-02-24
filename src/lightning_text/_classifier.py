import tempfile
from pathlib import Path
from typing import Any, Literal, TextIO, cast

import fasttext
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.utils.multiclass import (
    check_classification_targets,
    is_multilabel,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _fasttext_defaults as defaults
from ._dataset import Dataset
from ._fasttext import train_supervised
from ._fasttext_configuration import FastTextConfiguration
from ._fasttext_hyperparameters import SupervisedTrainingFastTextHyperparameters


class FastTextClassifier(
    SupervisedTrainingFastTextHyperparameters,
    FastTextConfiguration,
    ClassifierMixin,
    BaseEstimator,
):
    '''
    A scikit-learn estimator adapter for FastText supervised learning.

    In order to achieve compatibility with scikit-learn, there are some
    differences in the API compared to the original FastText API:

    - Classes are always numerical - 0 to n_classes - 1. Use a LabelEncoder or
        MultiLabelBinarizer to convert between class names and indices.
    - Predicting top k labels is not supported. Use the predict_proba method
        and implement the top k selection yourself.

    Note that the `fit()` method does not have a `sample_weight` parameter, as
    this is not supported by FastText. FastText's (hierarchical) softmax loss
    is said to handle unbalanced classes well. If the classes are balanced,
    negative sampling can be used as an alternative. See also:
    https://fasttext.cc/docs/en/faqs.html#why-is-the-hierarchical-softmax-slightly-worse-in-performance-than-the-full-softmax.
    '''

    label_encoder_: LabelEncoder | MultiLabelBinarizer | None = None
    '''
    The label encoder used to convert between class names and indices. This
    attribute is set by the user using the `fit()` method's `label_encoder`
    parameter and is used to train the underlying FastText classifier to use
    the original class names (the targets provided to the fit method are
    expected to be encoded by this encoder).

    Note that the predicted classes are not decoded back by the predict method
    to the class names known by this label encoder. This affects only the
    underlying FastText model's behavior, which can be useful if it were to be
    deployed without this scikit-learn adapter.
    '''

    feature_names_in_: np.ndarray[tuple[int], np.dtype[np.str_]] | None = None
    '''
    Names of features seen during fit. This should be only a single name, if
    set.
    '''

    n_features_in_: int | None = None
    '''
    Number of features seen during fit. This should always be 1, if set.
    '''

    classes_: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None
    '''
    The classes known to the classifier. This attribute is set by the fit
    method to the unique classes found in the target values for binary and 
    multi-class classification, or to the range of indices (0 to n_classes-1)
    for multi-label classification.
    '''

    model_: fasttext.FastText._FastText | None = None
    '''
    The trained FastText model.
    '''

    def __init__(
        self,
        *,
        lr: float = defaults.DEFAULT_LR_SUPERVISED,
        lr_update_rate: int = defaults.DEFAULT_LR_UPDATE_RATE,
        dim: int = defaults.DEFAULT_DIM,
        ws: int = defaults.DEFAULT_WS,
        epoch: int = defaults.DEFAULT_EPOCH,
        min_count: int = defaults.DEFAULT_MIN_COUNT_SUPERVISED,
        min_count_label: int = defaults.DEFAULT_MIN_COUNT_LABEL,
        minn: int = defaults.DEFAULT_MINN_SUPERVISED,
        maxn: int = defaults.DEFAULT_MAXN_SUPERVISED,
        neg: int = defaults.DEFAULT_NEG,
        word_ngrams: int = defaults.DEFAULT_WORD_NGRAMS,
        loss: Literal['softmax', 'ns', 'hs', 'ova'] = (
            defaults.DEFAULT_LOSS_SUPERVISED
        ),
        bucket: int = defaults.DEFAULT_BUCKET,
        t: float = defaults.DEFAULT_T,
        thread: int = defaults.DEFAULT_THREAD,
        label: str = defaults.DEFAULT_LABEL,
        verbose: int = defaults.DEFAULT_VERBOSE,
        pretrained_vectors: str | Path = defaults.DEFAULT_PRETRAINED_VECTORS,
        seed: int = defaults.DEFAULT_SEED,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.lr_update_rate = lr_update_rate
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.min_count = min_count
        self.min_count_label = min_count_label
        self.minn = minn
        self.maxn = maxn
        self.neg = neg
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.bucket = bucket
        self.t = t
        self.thread = thread
        self.label = label
        self.verbose = verbose
        self.pretrained_vectors = pretrained_vectors
        self.seed = seed

    def fit(
        self,
        X: np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.str_]],
        y: np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.int64]],
        label_encoder: LabelEncoder | MultiLabelBinarizer | None = None,
    ):
        '''
        Fit the model.

        Note that while multi-label training is supported, the FastText model
        always predicts a single label per sample (even if its loss is set to
        ova/one-vs-all). If multi-label classification is required, use the
        `MultiOutputClassifier` meta-estimator.

        Args:
            X: The training input samples. This should be in shape (n_samples,)
                or (n_samples, 1). It is recommended to preprocess the text
                data before passing it to this method. See the
                `preprocess_text` function in the `lightning_text` module.
                Pandas DataFrames are accepted as well.
            y: The target values. These are either class indices or multi-label
                binarized labels. Pandas series are accepted as well.
            label_encoder: The label encoder to use for encoding the labels as
                their original (string) representation. If None, the labels
                will be written as integers. This affects only the underlying
                FastText model's behavior.

        Returns:
            The fitted estimator (self).
        '''

        self.feature_names_in_ = None  # Necessary for validate_data().
        X, y = validate_data(
            self,
            X,
            y,
            dtype=X.dtype if hasattr(X, 'dtype') else np.str_,
            multi_output=is_multilabel(y),
            ensure_2d=hasattr(X, 'shape') and len(X.shape) == 2,
        )
        check_classification_targets(y)

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            y = cast(np.ndarray[tuple[int, int], np.dtype[np.int64]], y)
            self.classes_ = np.arange(y.shape[1])

        self.label_encoder_ = label_encoder

        dataset = Dataset(X.reshape(-1), y)
        temp_file = tempfile.NamedTemporaryFile(mode='xt', encoding='utf-8')
        with temp_file:
            dataset.write_to(
                cast(TextIO, temp_file),
                label_prefix=self.label,
                label_encoder=self.label_encoder_,
            )
            temp_file.flush()

            hyperparameters_and_configuration = self.get_params()
            self.model_ = train_supervised(
                input=temp_file.name,
                **hyperparameters_and_configuration,
            )

        return self

    def predict_proba(
        self,
        X: np.ndarray[tuple[int], np.dtype[np.str_]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        '''
        Predict the probabilities of each class for the input samples.

        Args:
            X: The input samples. These should be preprocessed text data (or at
                least not contain any newline characters).
                Pandas DataFrames are accepted as well.

        Returns:
            The predicted probabilities of each class for each sample. The
            returned array has shape (n_samples, n_classes), where n_samples is
            len(X) and n_classes is the number of classes known to the
            classifier. The class order is the same as the order in the
            `classes_` attribute (i.e. 0 to n_classes - 1).
        '''

        check_is_fitted(self)
        if not self.model_ or self.classes_ is None:  # For the type checker.
            raise NotFittedError(
                'Model not fitted, call the fit method first.',
            )

        X = validate_data(
            self,
            X,
            reset=False,
            dtype=X.dtype if hasattr(X, 'dtype') else np.str_,
            ensure_2d=hasattr(X, 'shape') and len(X.shape) == 2,
        )

        class_count = len(self.model_.get_labels())
        predicted_labels, predicted_probability = self.model_.predict(
            X.reshape(-1).tolist(),
            k=class_count,
        )

        label_prefix = self.label
        output_probabilities = np.zeros(
            (len(X), class_count),
            dtype=np.float32,
        )
        for sample_index, (sample_labels, sample_probabilities) in enumerate(
            zip(
                predicted_labels,
                predicted_probability,
            ),
        ):
            for label, probability in zip(sample_labels, sample_probabilities):
                trimmed_label = label[len(label_prefix):]
                if self.label_encoder_:
                    class_index = np.nonzero(
                        self.label_encoder_.classes_ == trimmed_label,
                    )[0][0]
                else:
                    class_index = int(trimmed_label)
                output_probabilities[sample_index, class_index] = probability

        return output_probabilities

    def predict(
        self,
        X: np.ndarray[tuple[int], np.dtype[np.str_]],
    ) -> np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.int64]]:
        '''
        Predict the class labels for the input samples.

        This adapter does not expose FastText's k parameter for returning top k
        predictions. If you need that functionality, use the predict_proba
        method and implement the top k selection yourself.

        Args:
            X: The input samples. These should be preprocessed text data (or at
                least not contain any newline characters).

        Returns:
            The predicted class labels for each sample. The shape of the
            returned array depends on the classifier's configuration.

            For binary and multi-class classification (`loss in ['softmax',
            'ns', 'hs']`), the returned array has shape (n_samples,), where
            n_samples is len(X). The values are the predicted classes (0 to
            n_classes - 1).

            For multi-label classification (`loss == 'ova'`), the returned array
            has shape (n_samples, n_classes), where n_samples is len(X) and
            n_classes is the number of classes known to the classifier. The
            values are binary indicators of the predicted classes.
        '''

        probabilities = self.predict_proba(X)
        if self.classes_ is None:  # For the type checker.
            raise NotFittedError(
                'Model not fitted, call the fit method first.',
            )

        if self.loss == 'ova':
            return (probabilities > 0.5).astype(np.int64)
        else:
            return self.classes_[np.argmax(probabilities, axis=1)]

    def save_model(self, path: Path | str) -> None:
        '''
        Save the underlying FastText model to a file. This does *not* save the
        configuration of the estimator, or the estimator itself.

        Use pickle for saving the estimator with its underlying model.

        Args:
            path: The file path where the model will be saved.
        '''

        if not self.model_:
            raise RuntimeError('Model not fitted, call the fit method first.')

        self.model_.save_model(str(path))

    def __sklearn_tags__(self):
        # See https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        tags = super().__sklearn_tags__()
        # Note: The following tag causes the check_estimator test to fail with
        # scikit-learn 1.6.1.
        tags.input_tags.one_d_array = True
        tags.input_tags.string = True
        return tags

    def __getstate__(self):
        '''
        Get the state of the estimator for pickling.

        Returns:
            The state of the estimator.
        '''

        state = super().__getstate__()

        # FastText model is not picklable, but can be saved to a file. Let's
        # exploit this.
        model = state.get('model_', None)
        state = {
            key: value
            for key, value in state.items()
            if key != 'model_'
        }
        if model is not None:
            temp_file = tempfile.NamedTemporaryFile(mode='xb')
            with temp_file:
                model.save_model(temp_file.name)
                temp_file.flush()
                state['model_bin'] = Path(temp_file.name).read_bytes()

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        '''
        Set the state of the estimator after unpickling.

        Args:
            state: The state of the estimator.
        '''

        # See __getstate__ for the explanation of this.
        model_bin = state.pop('model_bin', None)
        super().__setstate__(state)

        if model_bin is not None:
            temp_file = tempfile.NamedTemporaryFile(mode='xb')
            with temp_file:
                temp_file.write(model_bin)
                temp_file.flush()
                self.model_ = fasttext.load_model(temp_file.name)
