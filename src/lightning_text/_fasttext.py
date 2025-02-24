from pathlib import Path
from typing import Literal

import fasttext

from . import _fasttext_defaults as defaults


def tokenize(text: str) -> list[str]:
    '''
    Tokenize text into words.

    Args:
        text: The text to tokenize. This is a single example.

    Returns:
        The list of tokens.
    '''
    return fasttext.tokenize(text)


def load_model(path: str | Path) -> fasttext.FastText._FastText:
    '''
    Load a pretrained fastText model from a file. The model can be used for
    inference only (fastText does not support retraining/fine-tuning).

    Args:
        path: The model file path.

    Returns:
        The model object.
    '''
    return fasttext.load_model(str(path))


def train_supervised(
    input: str | Path,
    *,
    lr: float = defaults.DEFAULT_LR_SUPERVISED,
    lr_update_rate: int = defaults.DEFAULT_LR_UPDATE_RATE,
    dim: int = defaults.DEFAULT_DIM,
    ws: int = defaults.DEFAULT_WS,
    epoch: int = defaults.DEFAULT_EPOCH,
    min_count: int = defaults.DEFAULT_MIN_COUNT_SUPERVISED,
    min_count_label: int = defaults.DEFAULT_MIN_COUNT_LABEL,
    minn: int = 0,
    maxn: int = 0,
    neg: int = defaults.DEFAULT_NEG,
    word_ngrams: int = defaults.DEFAULT_WORD_NGRAMS,
    loss: Literal['softmax', 'ns', 'hs', 'ova'] = (
        defaults.DEFAULT_LOSS_SUPERVISED
    ),
    bucket: int = defaults.DEFAULT_BUCKET,
    thread: int = defaults.DEFAULT_THREAD,
    t: float = defaults.DEFAULT_T,
    label: str = defaults.DEFAULT_LABEL,
    verbose: int = defaults.DEFAULT_VERBOSE,
    pretrained_vectors: str | Path = defaults.DEFAULT_PRETRAINED_VECTORS,
    seed: int = defaults.DEFAULT_SEED,
    autotune_validation_file: str | Path = (
        defaults.DEFAULT_AUTOTUNE_VALIDATION_FILE
    ),
    autotune_metric: str = defaults.DEFAULT_AUTOTUNE_METRIC,
    autotune_predictions: int = defaults.DEFAULT_AUTOTUNE_PREDICTIONS,
    autotune_duration: int = defaults.DEFAULT_AUTOTUNE_DURATION,
    autotune_model_size: int | str = defaults.DEFAULT_AUTOTUNE_MODEL_SIZE,
) -> fasttext.FastText._FastText:
    '''
    Train a supervised model and return a model object.

    input must be a filepath. The input text does not need to be tokenized as
    per the tokenize function, but it must be preprocessed and encoded as UTF-8.
    You might want to consult standard preprocessing scripts such as
    tokenizer.perl mentioned at http://www.statmt.org/wmt07/baseline.html or use
    the preprocess_text function in this module.

    The input file must must contain at least one label per line. For an example
    consult the example datasets which are part of the FastText repository such
    as the dataset pulled by classification-example.sh.

    Args:
        input: The training data file path.
        lr: The learning rate.
        lr_update_rate: The rate of updates for the learning rate.
        dim: The dimension of word vectors.
        ws: The size of the context window.
        epoch: The number of training epochs.
        min_count: The minimal number of word occurrences.
        min_count_label: The minimal number of label occurrences.
        minn: The minimum length of char n-grams.
        maxn: The maximum length of char n-grams.
        neg: The number of negatives sampled.
        word_ngrams: The max length of word n-grams.
        loss: The loss function: 'softmax', 'ns', 'hs', or 'ova'.
        bucket: The number of buckets.
        thread: The number of threads.
        t: The sampling threshold.
        label: The label prefix.
        verbose: The verbosity level. Supported values are:
            - 0: no output
            - 1: print progress messages without updating progress during
                training
            - 2: print progress messages and update the progress during training
        pretrained_vectors: The pretrained vectors file path.
        seed: The random number generator seed. Use 0 for no seed, or non-zero
            value for reproducible results.
        autotune_validation_file: The validation data file path for autotuning.
            Providing a validation file will automatically enable autotuning.
        autotune_metric: The autotuning metric to optimize for. This can be the
            overall F1 score ('f1'), F1 score for a specific label (e.g.
            'f1:__label__my_label'), precision at a given recall (e.g.
            'precisionAtRecall:30' for precision at 30% recall), or recall at a
            given precision (e.g. 'recallAtPrecision:30' for recall at 30%). The
            precisionAtRecall and recallAtPrecision metrics can be optimized for
            a specific label as well (e.g.
            'precisionAtRecall:30:__label__my_label').

        autotune_predictions: The number of label predictions to optimize for.
        autotune_duration: The autotuning duration in seconds.
        autotune_model_size: The model size constraint for autotuning. If set,
            the model will be quantized to fit the given size. String values
            such as '2M' (2 megabytes) are accepted.
    '''

    return fasttext.train_supervised(
        input=str(input),
        lr=lr,
        lr_update_rate=lr_update_rate,
        dim=dim,
        ws=ws,
        epoch=epoch,
        min_count=min_count,
        minCountLabel=min_count_label,
        minn=minn,
        maxn=maxn,
        neg=neg,
        word_ngrams=word_ngrams,
        loss=loss,
        bucket=bucket,
        thread=thread,
        t=t,
        label=label,
        verbose=verbose,
        pretrained_vectors=str(pretrained_vectors),
        seed=seed,
        autotuneValidationFile=str(autotune_validation_file),
        autotuneMetric=autotune_metric,
        autotunePredictions=autotune_predictions,
        autotuneDuration=autotune_duration,
        autotuneModelSize=str(autotune_model_size),
    )


def train_unsupervised(
    input: str | Path,
    *,
    model: Literal['cbow', 'skipgram'] = defaults.DEFAULT_MODEL,
    lr: float = defaults.DEFAULT_LR_UNSUPERVISED,
    lr_update_rate: int = defaults.DEFAULT_LR_UPDATE_RATE,
    dim: int = defaults.DEFAULT_DIM,
    ws: int = defaults.DEFAULT_WS,
    epoch: int = defaults.DEFAULT_EPOCH,
    min_count: int = defaults.DEFAULT_MIN_COUNT_UNSUPERVISED,
    min_count_label: int = defaults.DEFAULT_MIN_COUNT_LABEL,
    minn: int = defaults.DEFAULT_MINN_UNSUPERVISED,
    maxn: int = defaults.DEFAULT_MAXN_UNSUPERVISED,
    neg: int = defaults.DEFAULT_NEG,
    word_ngrams: int = defaults.DEFAULT_WORD_NGRAMS,
    loss: Literal['ns', 'hs', 'ova'] = defaults.DEFAULT_LOSS_UNSUPERVISED,
    bucket: int = defaults.DEFAULT_BUCKET,
    thread: int = defaults.DEFAULT_THREAD,
    t: float = defaults.DEFAULT_T,
    label: str = defaults.DEFAULT_LABEL,
    verbose: int = defaults.DEFAULT_VERBOSE,
    pretrained_vectors: str | Path = defaults.DEFAULT_PRETRAINED_VECTORS,
) -> fasttext.FastText._FastText:
    '''
    Train an unsupervised model and return a model object.

    input must be a filepath. The input text does not need to be tokenized as
    per the tokenize function, but it must be preprocessed and encoded as UTF-8.
    You might want to consult standard preprocessing scripts such as
    tokenizer.perl mentioned at http://www.statmt.org/wmt07/baseline.html or use
    the preprocess_text function in this module.

    The input file must not contain any labels or use the specified label prefix
    unless it is ok for those words to be ignored. For an example consult the
    dataset pulled by the example script word-vector-example.sh, which is part
    of the FastText repository.

    Args:
        input: The training data file path.
        model: The training model: 'cbow' or 'skipgram'.
        lr: The learning rate.
        lr_update_rate: The rate of updates for the learning rate.
        dim: The dimension of word vectors.
        ws: The size of the context window.
        epoch: The number of training epochs.
        min_count: The minimal number of word occurrences.
        min_count_label: The minimal number of label occurrences.
        minn: The minimum length of char n-grams.
        maxn: The maximum length of char n-grams.
        neg: The number of negatives sampled.
        word_ngrams: The max length of word n-grams.
        loss: The loss function: 'ns', 'hs', or 'ova'.
        bucket: The number of buckets.
        thread: The number of threads.
        t: The sampling threshold.
        label: The label prefix.
        verbose: The verbosity level. Supported values are:
            - 0: no output
            - 1: print progress messages without updating progress during
                training
            - 2: print progress messages and update the progress during training
        pretrained_vectors: The pretrained vectors file path.

    Returns:
        The trained model object.
    '''

    return fasttext.train_unsupervised(
        input=str(input),
        model=model,
        lr=lr,
        lr_update_rate=lr_update_rate,
        dim=dim,
        ws=ws,
        epoch=epoch,
        min_count=min_count,
        minCountLabel=min_count_label,
        minn=minn,
        maxn=maxn,
        neg=neg,
        word_ngrams=word_ngrams,
        loss=loss,
        bucket=bucket,
        thread=thread,
        t=t,
        label=label,
        verbose=verbose,
        pretrained_vectors=str(pretrained_vectors),
    )
