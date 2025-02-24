from collections.abc import Collection
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    cast,
)

import numpy as np
import optuna
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._search import BaseSearchCV

from ._classifier import FastTextClassifier


@dataclass(frozen=True)
class Range[T: (int, float)]:
    '''
    A value range type for hyperparameter search.
    '''

    min: T
    '''
    The minimum value, inclusive.
    '''

    max: T
    '''
    The maximum value, inclusive.
    '''

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError('min must be less than or equal to max')


T = TypeVar('T', int, float)
ParameterRange = Range[T] | tuple[T, T]


class SupervisedTrainingHyperparametersSpace(TypedDict, total=False):
    '''
    A dictionary type for supervised training tunable parameters grid for
    hyperparameter search.
    '''

    lr: ParameterRange[float]
    '''
    The learning rate.
    '''

    lr_update_rate: ParameterRange[int]
    '''
    The rate of updates for the learning rate.
    '''

    dim: ParameterRange[int]
    '''
    The dimension of word vectors.
    '''

    ws: ParameterRange[int]
    '''
    The size of the context window.
    '''

    epoch: ParameterRange[int]
    '''
    The number of training epochs.
    '''

    min_count: ParameterRange[int]
    '''
    The minimal number of word occurrences.
    '''

    min_count_label: ParameterRange[int]
    '''
    The minimal number of label occurrences.
    '''

    minn: ParameterRange[int]
    '''
    The minimum length of char n-grams.
    '''

    maxn: ParameterRange[int]
    '''
    The maximum length of char n-grams.
    '''

    neg: ParameterRange[int]
    '''
    The number of negatives sampled.
    '''

    word_ngrams: ParameterRange[int]
    '''
    The max length of word n-grams.
    '''

    loss: Collection[Literal['softmax', 'ns', 'hs', 'ova']]
    '''
    The loss function: 'softmax', 'ns' (negative sampling), 'hs' (hierarchical
    softmax), or 'ova' (one over all).
    '''

    bucket: ParameterRange[int]
    '''
    The number of buckets.
    '''

    t: ParameterRange[float]
    '''
    The sampling threshold.
    '''

    post_process_params: Callable[
        [
            'SupervisedTrainingHyperparametersSpace',
            dict[str, Any],
        ],
        dict[str, Any],
    ]
    '''
    A function to post-process the selected hyperparameters to ensure they are
    valid.

    See also:
        default_post_process_hyperparameters
    '''


# See https://github.com/facebookresearch/fastText/blob/main/src/autotune.cc#L136-L182
def default_post_process_hyperparameters(
    space: SupervisedTrainingHyperparametersSpace,
    params: dict[str, Any],
) -> dict[str, Any]:
    '''
    Post-processes the hyperparameters to ensure they are valid. This is the
    default value of the 'post_process_params' key in the
    DEFAULT_SUPERVISED_TRAINING_HYPERPARAMETERS_SPACE dictionary.

    Args:
        space: The hyperparameters space.
        params: The hyperparameters to post-process.

    Returns:
        The post-processed hyperparameters.
    '''

    result = params.copy()

    minn_choices = [0, 2, 3]
    if 'minn' in result and 0 <= result['minn'] <= len(minn_choices):
        result['minn'] = minn_choices[result['minn']]

    if 'minn' not in result or result['minn'] == 0:
        result['maxn'] = 0
    else:
        result['maxn'] = result['minn'] + 3

    if (
        'wordngrams' in result and
        result['wordngrams'] <= 1 and
        result['maxn'] == 0
    ):
        result['bucket'] = 0

    return result


# The default hyperparameters space for supervised training, as suggested by
# https://github.com/facebookresearch/fastText/blob/main/src/autotune.cc#L136-L182
DEFAULT_SUPERVISED_TRAINING_HYPERPARAMETERS_SPACE: \
    SupervisedTrainingHyperparametersSpace = {
        'epoch': Range(1, 100),
        'lr': Range(0.01, 5.0),
        'dim': Range(1, 1000),
        'word_ngrams': Range(1, 5),
        # 'dsub': Range(1, 4),  # Use the quantize method on the trained model
        'minn': Range(0, 2),
        # 'maxn': Range(0, 0),  # Derived from 'minn' by post-processor
        'bucket': Range(10_000, 10_000_000),
        'post_process_params': default_post_process_hyperparameters,
    }


class ParameterCandidateEvaluator(Protocol):
    '''
    A protocol for evaluating a candidate parameter set, used by BaseSearchCV.

    The returned dictionary contains all metrics for so-far completed trials in
    chronological order, i.e. the last item is the most recent trial.

    See OptunaSearchCV.cv_results_ for documentation of the returned dictionary.
    '''

    def __call__(
        self,
        candidate_params: Iterable[dict[str, Any]],
        cv: BaseCrossValidator | None = None,
        more_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


class OptunaSearchCV(BaseSearchCV):
    '''
    A hyperparameter search for FastTextClassifier using Optuna (see
    https://optuna.org/).

    Example usage:
    ```
    from typing import Any, cast

    import optuna
    from sklearn.metrics import fbeta_score, make_scorer
    from sklearn.model_selection import (BaseCrossValidator,
                                         RepeatedStratifiedKFold)
    from lightning_text import (Dataset, FastTextClassifier,
                               OptunaSearchCV, Range,
                               SupervisedTrainingHyperparametersSpace)

    dataset, label_encoder = Dataset.from_file('dataset.txt')
    params_space: SupervisedTrainingHyperparametersSpace = {
        'epoch': Range(1, 100),
    }
    estimator = FastTextClassifier()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    def metrics_to_optuna_goals(metrics: dict[str, Any]) -> float:
        last_mean_fbeta = metrics['mean_test_score'][-1]
        return last_mean_fbeta

    study = optuna.create_study(direction='maximize')
    search = OptunaSearchCV(
        estimator=estimator,
        study=study,
        hyperparameters_space=params_space,
        n_iter=64,
        scoring=make_scorer(fbeta_score, pos_label=1, beta=1),
        optuna_metrics_exporter=metrics_to_optuna_goals,
        n_jobs=4,
        refit=True, # or metric name for multi-metric scoring
        cv=cast(BaseCrossValidator, cv),
        show_progress_bar=True,
    )

    search.fit(dataset.X, dataset.y)

    print(search.best_params_)
    print(search.best_score_)
    search.best_estimator_.save_model('best_model.bin')
    '''

    study: optuna.study.Study
    '''
    The Optuna study to use for the search.
    '''

    hyperparameters_space: SupervisedTrainingHyperparametersSpace
    '''
    The hyperparameters space to search in.
    '''

    n_iter: int | None
    '''
    The number of iterations to run the search for. Use None for unlimited - the
    search will stop when the timeout is reached or a termination signal (e.g.
    Ctrl+c) is received.
    '''

    timeout: timedelta | float | None
    '''
    The maximum time to run the search for. Use None for unlimited - the search
    will stop when the n_iter is reached or a termination signal (e.g. Ctrl+c)
    is received. If a float is provided, it is interpreted as the number of
    seconds.
    '''

    optuna_metrics_exporter: Callable[
        [dict[str, Any]],
        float | Sequence[float],
    ]
    '''
    A function to convert the metrics dictionary to a value(s) to optimize for.
    The function should return a single value if the search is for a single
    metric, or a sequence of values (always in the same order) if the search is
    for multiple metrics.

    See `OptunaSearchCV.cv_results_` for documentation of the metrics
    dictionary.
    '''

    show_progress_bar: bool
    '''
    Whether to show a progress bar during the search. The progress bar is
    provided by the optuna library.
    '''

    cv_results_: dict[str, Any]
    '''
    Dictionary containing the metrics for all evaluated candidates, containing
    the following keys, with values being either numpy arrays or python lists:
    - `'mean_fit_time'`: The mean time taken to fit the estimator on the
      training data for each hyperparameter candidate. The mean is taken across
      all folds.
    - `'std_fit_time'`: The standard deviation of the time taken to fit the
      estimator on the training data for each hyperparameter candidate. The
      standard deviation is taken across all folds.
    - `'mean_score_time'`: The mean time taken to score the estimator on the
      test data for each hyperparameter candidate. The mean is taken across all
      folds.
    - `'std_score_time'`: The standard deviation of the time taken to score the
      estimator on the test data for each hyperparameter candidate. The standard
      deviation is taken across all folds.
    - `'params'`: The hyperparameters for each hyperparameter candidate. Each
      hyperparameter candidate is a dictionary containing the hyperparameters as
      `{param_name: param_value}`.
    - `f'param_{param_name}'`: The hyperparameter values for the param_name
      hyperparameter for each candidate.
    - `f'split{i}_test_score'`: The score for each fold of the cross-validation
      for each hyperparameter candidate. The score is computed using the
      scoring function. This key is present only if the scoring function returns
      a single value.
    - `f'split{i}_test_{metric}'`: The score for each fold of the
      cross-validation for each hyperparameter candidate. The score is computed
      using the scoring function. This key is present only if the scoring
      function returns a dictionary and exists for every metric in the
      dictionary.
    - `'mean_test_score'`: The mean score across all folds
      (`f'split{i}_test_score'`) for each hyperparameter candidate. This key is
      present only if the scoring function returns a single value.
    - `'std_test_score'`: The standard deviation of the score across all folds
      (`f'split{i}_test_score'`) for each hyperparameter candidate. This key is
        present only if the scoring function returns a single value.
    - `'rank_test_score'`: The rank of each hyperparameter candidate based on
      the mean scores (`'mean_test_score'`). The candidate with the best score
      is assigned rank 1. This key is present only if the scoring function
      returns a single value.
    - `f'mean_test_{metric}'`: The mean score across all folds
      (`f'split{i}_test_{metric}'`) for each hyperparameter candidate. This key
      is present only if the scoring function returns a dictionary and exists
      for every metric in the dictionary.
    - `f'std_test_{metric}'`: The standard deviation of the score across all
      folds (`f'split{i}_test_{metric}'`) for each hyperparameter candidate.
      This key is present only if the scoring function returns a dictionary and
      exists for every metric in the dictionary.
    - `f'rank_test_{metric}'`: The rank of each hyperparameter candidate based
      on the mean scores (`f'mean_test_{metric}`). The candidate with the best
      score is assigned rank 1. This key is present only if the scoring function
      returns a dictionary and exists for every metric in the dictionary.
    - 'f`split{i}_train_score'`: The score for each fold of the cross-validation
      for each hyperparameter candidate. The score is computed using the
      scoring function. This key is present only if return_train_score is True
      and the scoring function returns a single value.
    - `f'split{i}_train_{metric}'`: The score for each fold of the
      cross-validation for each hyperparameter candidate. The score is computed
      using the scoring function. This key is present only if return_train_score
      is True and the scoring function returns a dictionary. This key exists for
      every metric in the dictionary returned by the scoring function.
    - `'mean_train_score'`: The mean score on training data across all folds
      (`f'split{i}_train_score'`) for each hyperparameter candidate. This key is
      present only if return_train_score is True and the scoring function
      returns a single value.
    - `'std_train_score'`: The standard deviation of the score on training data
      across all folds (`f'split{i}_train_score'`) for each hyperparameter
      candidate. This key is present only if return_train_score is True and the
      scoring function returns a single value.
    - `f'mean_train_{metric}'`: The mean score on training data across all folds
      (`f'split{i}_train_{metric}'`) for each hyperparameter candidate. This key
      is present only if return_train_score is True and the scoring function
      returns a dictionary. This key exists for every metric in the dictionary
      returned by the scoring function.
    - `f'std_train_{metric}'`: The standard deviation of the score on training
      data across all folds (`f'split{i}_train_{metric}'`) for each
      hyperparameter candidate. This key is present only if return_train_score
      is True and the scoring function returns a dictionary. This key exists for
      every metric in the dictionary returned by the scoring function.
    '''

    best_index_: int
    '''
    The index of the best hyperparameters in the
    `cv_results_['params']` array, and its related metrics in the
    `cv_results_` dictionary.

    Note that this attribute is set only if refit is not False or scoring
    returns a single value.
    '''

    best_params_: dict[str, Any]
    '''
    The best hyperparameters found during the search. This field is equal to
    `cv_results_['params'][best_index_]`.

    Note that this attribute is set only if refit is not False or scoring
    returns a single value.
    '''

    best_score_: float
    '''
    The best score found during the search. This field is equal to:
    - `cv_results_['mean_test_score'][best_index_]` if scoring returns a single
      value,
    - `cv_results_[f'mean_test_{refit}'][best_index_]` if scoring returns a
      dictionary.
    
    Note that this attribute is set only if refit is not False or scoring
    returns a single value. This attribute is never available if refit is a
    function.
    '''

    refit_time_: float
    '''
    The time taken to refit the best model after the search is complete. This
    is not available if refit is False.
    '''

    multimetric_: bool
    '''
    Whether or not the scoring function returns a dictionary (even if the
    dictionary contains a single value).
    '''

    best_estimator_: FastTextClassifier
    '''
    The estimator refitted with the best hyperparameters found during the
    search.

    This field is available only if refit is not False.
    '''

    def __init__(
        self,
        estimator: FastTextClassifier,
        study: optuna.study.Study,
        hyperparameters_space: SupervisedTrainingHyperparametersSpace,
        n_iter: int | None = None,
        timeout: float | None = None,
        *,
        scoring: (
            Callable[
                [
                    FastTextClassifier,
                    np.ndarray[tuple[int], np.dtype[np.str_]],
                    np.ndarray[
                        tuple[int] | tuple[int, int],
                        np.dtype[np.int64],
                    ],
                ],
                dict[str, int | float | np.float64],
            ]
        ),
        optuna_metrics_exporter: Callable[
            [dict[str, Any]],
            float | Sequence[float],
        ],
        n_jobs: int | None = None,
        refit: bool | str | Callable = True,
        cv: (
            int |
            BaseCrossValidator |
            Iterable[tuple[
                np.ndarray[tuple[int], np.dtype[np.int64]],
                np.ndarray[tuple[int], np.dtype[np.int64]],
            ]] |
            None
        ) = None,
        show_progress_bar: bool = False,
        verbose: int = 0,
        pre_dispatch: int | str = '2*n_jobs',
        error_score: Literal['raise'] | int | float = np.nan,
        return_train_score: bool = False,
    ):
        '''
        Create a new OptunaSearchCV instance for hyperparameter search.

        Args:
            estimator: The estimator to use for training.
            study: The Optuna study to use for the search.
            hyperparameters_space: The hyperparameters space to search in .
            n_iter: The number of iterations to run the search for . Use None for
                unlimited - the search will stop when the timeout is reached or
                a termination signal(e.g. Ctrl+c) is received.
            timeout: The maximum time to run the search for . Use None for
                unlimited - the search will stop when the n_iter is reached or a
                termination signal(e.g. Ctrl+c) is received. If a float is
                provided, it is interpreted as the number of seconds.
            scoring: The scoring function to use for evaluating the candidates.
                Use the metrics_to_scorer function to convert a metrics callback
                to a scoring function.
            optuna_metrics_exporter: A function to convert the metrics
                dictionary to a value(s) to optimize for . The function should
                return a single value if the search is for a single metric, or a
                sequence of values(always in the same order) if the search is
                for multiple metrics.
            n_jobs: The number of jobs to run in parallel. Use None for 1 job,
                unless the search is run in a joblib.parallel_backend context.
                Note that FastText will attempt to utilize all available CPU
                cores regardless of this parameter(unless configured otherwise
                using the 'thread' parameter), so setting this parameter to a
                value greater than 1 may not result in a significant speedup.
            refit: Whether to refit the best estimator with the entire dataset
                after the search is complete. If a string is provided, it is
                interpreted as the name of the metric to use for refitting. If a
                callable is provided, it is called with the metrics dictionary
                and should return a value to use for refitting.
                It is usually preferable to provide a string or a callable over
                a boolean value, because metrics always return a dictionary,
                therefore specifying the metric for picking the best model is
                always necessary(unless a callback returning a single numerical
                value is passed to the scoring parameter).
            cv: The cross-validation strategy to use. Use an integer to specify
                the number of folds, a BaseCrossValidator instance to specify a
                custom cross-validator(e.g. RepeatedStratifiedKFold), or an
                iterable of(X, y) pairs of indices to specify the training and
                validation sets directly. Use None to use the default 5-fold
                cross-validation.
            show_progress_bar: Whether to show a progress bar during the search.
                The progress bar is provided by the optuna library.
            verbose: The scikit-learn's BaseSearchCV's verbosity level.
                Supported values are:
                - 0: no output
                - 1: the number of folds and parameter candidates in a single
                     search batch is displayed(this hyperparameter search
                     always evaluates a single candidate at a time).
                - 2: the computation time for each fold and candidate is
                     displayed.
                - 3: the score is also displayed.
                - 4: fold and candidate indexes are also displayed together with
                     the starting time of the computation.
            pre_dispatch: The number of training jobs dispatched in parallel.
                The default value is '2*n_jobs', which means that the number of
                jobs dispatched in parallel is twice the number of jobs
                specified by the n_jobs parameter.
            error_score: The value to assign to the score if an error occurs
                while evaluating a candidate. Use 'raise' to raise an exception
                in case of an error, or a numerical value to assign that value
                to the score.
            return_train_score: Whether to return the training scores in the
                cv_results_ attribute.
        '''

        # BaseSearchCV's __init__ method is annotated as abstract, that's why
        # there as a type: ignore comment.
        super().__init__(  # type: ignore
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=cast(bool, refit),
            cv=cv,
            verbose=verbose,
            pre_dispatch=str(pre_dispatch),
            error_score=error_score,
            return_train_score=return_train_score,
        )

        self.study = study
        self.hyperparameters_space = hyperparameters_space
        self.n_iter = n_iter
        self.timeout = timeout
        self.optuna_metrics_exporter = optuna_metrics_exporter
        self.show_progress_bar = show_progress_bar

    def _run_search(
        self,
        evaluate_candidates: ParameterCandidateEvaluator,
    ) -> None:
        '''
        Run the hyperparameter search using Optuna. This method is called by
        fit method, do not call it directly.

        Args:
            evaluate_candidates: The function to evaluate the hyperparameter
                candidates. It returns a dictionary containing the metrics for
                the so-far evaluated candidates in chronological order.
        '''

        def objective(trial: optuna.trial.Trial) -> float | Sequence[float]:
            hyperparameters = {}
            for param_name, range in self.hyperparameters_space.items():
                if param_name == 'post_process_params':
                    continue
                elif param_name == 'loss':
                    hyperparameters[param_name] = cast(
                        Literal['softmax', 'ns', 'hs', 'ova'],
                        trial.suggest_categorical(
                            param_name,
                            list(cast(Collection[str], range)),
                        ),
                    )

                if isinstance(range, tuple):
                    range = Range(*range)
                range = cast(Range[int | float], range)

                if isinstance(range.min, int) and isinstance(range.max, int):
                    hyperparameters[param_name] = trial.suggest_int(
                        param_name,
                        range.min,
                        range.max,
                    )
                else:
                    hyperparameters[param_name] = trial.suggest_float(
                        param_name,
                        range.min,
                        range.max,
                    )

            if 'post_process_params' in self.hyperparameters_space:
                hyperparameters = (
                    self.hyperparameters_space['post_process_params'](
                        self.hyperparameters_space,
                        hyperparameters,
                    )
                )

            metrics = evaluate_candidates([hyperparameters])

            return self.optuna_metrics_exporter(metrics)

        timeout = self.timeout.total_seconds() if isinstance(
            self.timeout,
            timedelta,
        ) else self.timeout

        # Note: we cannot use the n_jobs parameter here, because BaseSearchCV
        # will already start a Parallel job (even when its n_jobs==1), and we
        # cannot nest Parallel jobs.
        self.study.optimize(
            objective,
            n_trials=self.n_iter,
            timeout=timeout,
            show_progress_bar=self.show_progress_bar,
        )
