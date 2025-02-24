from typing import Literal

from . import _fasttext_defaults as defaults


class SupervisedTrainingFastTextHyperparameters:
    '''
    A dictionary type for supervised training hyperparameters, primarily for
    hyperparameter search.
    '''

    lr: float = defaults.DEFAULT_LR_SUPERVISED
    '''
    The learning rate.
    '''

    lr_update_rate: int = defaults.DEFAULT_LR_UPDATE_RATE
    '''
    The rate of updates for the learning rate.
    '''

    dim: int = defaults.DEFAULT_DIM
    '''
    The dimension of word vectors.
    '''

    ws: int = defaults.DEFAULT_WS
    '''
    The size of the context window.
    '''

    epoch: int = defaults.DEFAULT_EPOCH
    '''
    The number of training epochs.
    '''

    min_count: int = defaults.DEFAULT_MIN_COUNT_SUPERVISED
    '''
    The minimal number of word occurrences.
    '''

    min_count_label: int = defaults.DEFAULT_MIN_COUNT_LABEL
    '''
    The minimal number of label occurrences.
    '''

    minn: int = defaults.DEFAULT_MINN_SUPERVISED
    '''
    The minimum length of char n-grams.
    '''

    maxn: int = defaults.DEFAULT_MAXN_SUPERVISED
    '''
    The maximum length of char n-grams.
    '''

    neg: int = defaults.DEFAULT_NEG
    '''
    The number of negatives sampled.
    '''

    word_ngrams: int = defaults.DEFAULT_WORD_NGRAMS
    '''
    The max length of word n-grams.
    '''

    loss: Literal['softmax', 'ns', 'hs', 'ova'] = (
        defaults.DEFAULT_LOSS_SUPERVISED
    )
    '''
    The loss function: 'softmax', 'ns' (negative sampling), 'hs' (hierarchical
    softmax), or 'ova' (one over all).
    '''

    bucket: int = defaults.DEFAULT_BUCKET
    '''
    The number of buckets.
    '''

    t: float = defaults.DEFAULT_T
    '''
    The sampling threshold.
    '''
