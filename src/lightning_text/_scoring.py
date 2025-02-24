from statistics import harmonic_mean
from typing import Any, Iterable

import numpy as np


def get_fold_scores_per_candidate(
    cv_results: dict[str, Any],
) -> dict[str, np.ndarray[tuple[int, int], np.dtype[np.float64]]]:
    '''
    Extracts the scores of each fold for each candidate from the cv_results (see
    `OptunaSearchCV.cv_results_`) from hyperparameter search using
    cross-validation.

    The returned dictionary has the following structure:
    {
        'metric_name': np.ndarray[
            (n_candidates, n_folds),
            np.float64,
        ],
        ...
    }

    This function is meant to be used if more advanced analysis than the mean
    and standard deviation of collected metrics is needed for individual
    candidates (e.g. to calculate the confidence interval of the mean, use
    harmonic mean, calculate stability score, etc.).

    Args:
        cv_results: Results of the cross-validation search.

    Returns:
        Dictionary with scores for each fold for each candidate.
    '''

    metric_names = {  # f'{'train' or 'test'}_{metric_name}'
        key[key.index('_') + 1:]
        for key in cv_results.keys()
        if key.startswith('split')
    }
    if not metric_names:
        raise ValueError('No metrics found in cv_results.')

    result = {}
    for metric_name in metric_names:
        split_scores = np.array([
            cv_results[f'split{i}_{metric_name}']
            for i in range(len(cv_results))
            if f'split{i}_{metric_name}' in cv_results
        ])
        result[metric_name] = split_scores.T

    return result


def robust_cv_score(
    scores: Iterable[float] | np.ndarray[tuple[int], np.dtype[np.float64]],
) -> float:
    '''
    Harmonic mean puts more weight on lower scores, naturally penalizing high
    variance.

    This function is meant to be used as a scoring function for hyperparameter
    optimization, where the goal is to find hyperparameters that perform well
    on average and are stable across folds.

    Use this function to produce a score for the hyperparameter search to
    optimize for.

    Example:
    ```
    def cv_results_to_optuna_goal(cv_results: dict[str, Any]) -> float:
        # group fold scores by candidate
        fold_scores = get_fold_scores_per_candidate(cv_results)

        # get fold F_beta scores for the last evaluated candidate
        candidate_fbeta_scores = fold_scores['test_fbeta'][-1]

        # return the robust score
        return robust_cv_score(candidate_fbeta_scores)

    ...

    search = OptunaSearchCV(
        ...
        optuna_metrics_exporter=cv_results_to_optuna_goal,
        ...
    )
    ```

    Args:
        scores: Scores from cross-validation folds.

    Returns:
        Robust score (harmonic mean of the scores).
    '''

    return harmonic_mean(scores)


def penalized_cv_score(
    scores: list[float] | np.ndarray[tuple[int], np.dtype[np.float64]],
    penalty_weight: float = 1.0,
) -> np.float64:
    '''
    Explicitly penalize standard deviation of scores.

    This function is meant to be used as a scoring function for hyperparameter
    optimization, where the goal is to find hyperparameters that perform well
    on average and are stable across folds.

    Use this function to produce a score for the hyperparameter search to
    optimize for.

    Note that the penalty_weight should be set to a value that makes sense for
    the specific problem and the importance of the stability of the model. Try
    multiple values to see how they affect the results.

    Example:
    ```
    def cv_results_to_optuna_goal(cv_results: dict[str, Any]) -> float:
        # group fold scores by candidate
        fold_scores = get_fold_scores_per_candidate(cv_results)

        # get fold F_beta scores for the last evaluated candidate
        candidate_fbeta_scores = fold_scores['test_fbeta'][-1]

        # return the penalized score
        return penalized_cv_score(
            candidate_fbeta_scores,
            penalty_weight=1.0,
        )

    ...

    search = OptunaSearchCV(
        ...
        optuna_metrics_exporter=cv_results_to_optuna_goal,
        ...
    )
    ```

    Args:
        scores: Scores from cross-validation folds.
        penalty_weight: Weight of the standard deviation penalty.

    Returns:
        Mean score penalized by the standard deviation.
    '''
    return np.mean(scores) - penalty_weight * np.std(scores)


def stability_score(
    scores: np.ndarray[tuple[int], np.dtype[np.float64]],
    threshold=0.1,
) -> np.float64:
    '''
    Measure how many scores are within threshold of mean and return the ratio.

    This function is meant to be used as a scoring function for hyperparameter
    optimization, where the goal is to find hyperparameters that perform well
    on average and are stable across folds.

    Use this function to produce a score for the hyperparameter search to
    optimize for.

    As this function does not take into account the actual values of the scores,
    it is recommended to use it in combination with another scoring function
    that does (e.g. `robust_cv_score`) and optimize for multiple objectives
    (with optimal weights, ideally).

    Note that the threshold should be set to a value that makes sense for the
    specific problem and the importance of the stability of the model. Try
    multiple values to see how they affect the results.

    Example:
    ```
    def cv_results_to_optuna_goal(cv_results: dict[str, Any]) -> float:
        # group fold scores by candidate
        fold_scores = get_fold_scores_per_candidate(cv_results)

        # get fold F_beta scores for the last evaluated candidate
        candidate_fbeta_scores = fold_scores['test_fbeta'][-1]

        # calculate the stability score
        stability = stability_score(candidate_fbeta_scores)

        # get last candidate's mean F_beta score
        mean_fbeta = np.mean(candidate_fbeta_scores)
        # another way: mean_fbeta = cv_results['mean_test_fbeta'][-1]

        # optimize for both metrics
        return mean_fbeta, stability

    ...

    search = OptunaSearchCV(
        ...
        optuna_metrics_exporter=cv_results_to_optuna_goal,
        ...
    )
    ```

    Args:
        scores: Scores from cross-validation folds.
        threshold: Threshold for the scores to be considered stable.

    Returns:
        Ratio of scores within threshold of the mean.
    '''

    mean = np.mean(scores)
    within_threshold = np.sum(np.abs(scores - mean) <= threshold)
    return within_threshold / len(scores)
