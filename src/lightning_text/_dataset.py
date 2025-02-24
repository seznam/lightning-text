from itertools import chain
from pathlib import Path
from typing import Literal, TextIO, cast

import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from ._fasttext_defaults import DEFAULT_LABEL


class Dataset:
    '''
    Representation of a FastText dataset compatible with scikit-learn.
    '''

    X: np.ndarray[tuple[int], np.dtype[np.str_]]
    '''
    The text data in the dataset.
    '''

    y: np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.int64]]
    '''
    The target labels in the dataset, encoded as integers, binarized in case
    of multi-label data.
    '''

    def __init__(
        self,
        X: np.ndarray[tuple[int], np.dtype[np.str_]],
        y: np.ndarray[tuple[int] | tuple[int, int], np.dtype[np.int64]],
    ) -> None:
        if len(X) != len(y):
            raise ValueError('data and target must have the same length.')

        self.X = X
        self.y = y

    def write_to(
        self,
        file: TextIO,
        label_prefix: str = DEFAULT_LABEL,
        label_encoder: LabelEncoder | MultiLabelBinarizer | None = None,
    ) -> None:
        '''
        Write the dataset to a file.

        Args:
            file: The file to write the dataset to.
            label_prefix: The label prefix used by FastText to distinguish
                between data labels and regular text.
            label_encoder: The label encoder to use for encoding the labels as
                their original (string) representation. If None, the labels
                will be written as integers.
        '''

        for text, labels in zip(self.X, self.y):
            if label_encoder:
                encoded_labels = (
                    f'{label_prefix}{label}'
                    for label in np.array(label_encoder.inverse_transform(
                        np.array([labels])
                    )).flatten()
                )
            elif isinstance(labels, np.ndarray):
                encoded_labels = (
                    f'{label_prefix}{label_id}'
                    for label_id, label_is_set in enumerate(labels)
                    if label_is_set != 0
                )
            else:
                encoded_labels = (
                    f'{label_prefix}{label}'
                    for label in (labels,)
                )
            serialized_entry = ' '.join(chain(encoded_labels, (text,)))
            file.write(serialized_entry + '\n')

    @classmethod
    def load_from(
        cls,
        file: TextIO,
        label_prefix: str = DEFAULT_LABEL,
    ) -> 'tuple[Dataset, LabelEncoder | MultiLabelBinarizer]':
        '''
        Load the dataset from a file.

        Args:
            file: The file to load the dataset from.
            label_prefix: The label prefix.

        Returns:
            The loaded dataset.
        '''

        X = []
        labels = []
        is_multi_label = False

        for line in file:
            parts = line.strip().split()
            sample_labels = set()
            text_parts = []
            for part in parts:
                if part.startswith(label_prefix):
                    sample_labels.add(part[len(label_prefix):])
                else:
                    text_parts.append(part)
            text = ' '.join(text_parts)
            sample_labels = list(sample_labels)
            if len(sample_labels) > 1:
                is_multi_label = True
            X.append(text)
            labels.append(sample_labels)

        if is_multi_label:
            label_encoder = MultiLabelBinarizer()
            y = label_encoder.fit_transform(labels)
        else:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)

        instance = cls(
            np.array(X, dtype=np.str_),
            cast(np.ndarray, y),
        )

        return instance, label_encoder

    def to_file(
        self,
        file_path: str | Path,
        mode: Literal['w', 'a', 'x', 'wt', 'at', 'xt'] = 'w',
        label_prefix: str = DEFAULT_LABEL,
        label_encoder: LabelEncoder | MultiLabelBinarizer | None = None,
    ) -> None:
        '''
        Write the dataset to a file.

        Args:
            file_path: The path to the output file.
            mode: The mode to open the file with.
            label_prefix: The prefix used for labels in the file, used by
                FastText to distinguish between data labels and regular text.
            label_encoder: The label encoder to use for encoding the labels as
                their original (string) representation. If None, the labels
                will be written as integers.
        '''

        with open(file_path, mode, encoding='utf-8') as file:
            self.write_to(file, label_prefix, label_encoder)

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        label_prefix: str = DEFAULT_LABEL,
    ) -> 'tuple[Dataset, LabelEncoder | MultiLabelBinarizer]':
        '''
        Load a Dataset from a file.

        Args:
            file_path: The path to the input file.
            label_prefix: The prefix used for labels in the file.

        Returns:
            A new Dataset instance.
        '''

        with open(file_path, encoding='utf-8') as file:
            return cls.load_from(file, label_prefix)
