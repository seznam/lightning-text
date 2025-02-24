from pathlib import Path

from . import _fasttext_defaults as defaults


class FastTextConfiguration:
    '''
    A dictionary type for fastText configuration.
    '''

    thread: int = defaults.DEFAULT_THREAD
    '''
    The number of threads to use for training.
    '''

    label: str = defaults.DEFAULT_LABEL
    '''
    Prefix used to identify labels in the internally generated training file.
    '''

    verbose: int = defaults.DEFAULT_VERBOSE
    '''
    FastText's verbosity level. Supported values are:
    - 0: no output
    - 1: print progress messages without updating progress meter during training
    - 2: print progress messages and update the progress meter during training
    '''

    pretrained_vectors: str | Path = defaults.DEFAULT_PRETRAINED_VECTORS
    '''
    Path to pretrained word vectors to use. Set to empty string if embedding
    should be trained from scratch.
    '''

    seed: int = defaults.DEFAULT_SEED
    '''
    Random seed to use for FastText training. Use 0 for no seed, or non-zero
    value for reproducible results.
    '''
