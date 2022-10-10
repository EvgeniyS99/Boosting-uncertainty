from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_array
from sklearn.utils.validation import indexable

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-16) -> float:
    output = (np.abs(y_true - y_pred)) / np.clip(np.abs(y_true), a_min=eps, a_max=None)
    return np.mean(output)

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-16) -> float:
    output = 2 * np.abs(y_true - y_pred) / np.clip(np.abs(y_true) + np.abs(y_pred), a_min=eps, a_max=None)
    return np.mean(output)

def wape(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-16) -> float:
    output = np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
    return output

def bias(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-16) -> float:
    output = np.sum((y_pred - y_true)) / np.sum(y_true)
    return output

class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum groups for a single training set.
    test_size : int, default=None
        Number of groups in test
    gap : int, default=0
        Number of groups between train and test sets
    Examples
    --------
    >>> import numpy as np
    >>> groups = np.array(['a', 'a', 'a', 'a', 'a', 'a',\
                    'b', 'b', 'b', 'b', 'b',\
                    'c', 'c', 'c', 'c',\
                    'd', 'd', 'd',
                    'e', 'e', 'e'])
    >>> splitter = GroupTimeSeriesSplit(n_splits=3, max_train_size=2, gap=1)
    >>> for i, (train_idx, test_idx) in enumerate(
    ...     splitter.split(groups, groups=groups)):
    ...     print(f"Split: {i + 1}")
    ...     print(f"Train idx: {train_idx}, test idx: {test_idx}")
    ...     print(f"Train groups: {groups[train_idx]},
                    test groups: {groups[test_idx]}\n")
    Split: 1
    Train idx: [0 1 2 3 4 5], test idx: [11 12 13 14]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a'], test groups: ['c' 'c' 'c' 'c']

    Split: 2
    Train idx: [ 0  1  2  3  4  5  6  7  8  9 10], test idx: [15 16 17]
    Train groups: ['a' 'a' 'a' 'a' 'a' 'a' 'b' 'b' 'b' 'b' 'b'],
    test groups: ['d' 'd' 'd']

    Split: 3
    Train idx: [ 6  7  8  9 10 11 12 13 14], test idx: [18 19 20]
    Train groups: ['b' 'b' 'b' 'b' 'b' 'c' 'c' 'c' 'c'],
    test groups: ['e' 'e' 'e']
    """

    def __init__(self, n_splits=5, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        unique_groups, ind = np.unique(groups, return_index=True)
        n_groups = _num_samples(unique_groups)
        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f"than the number of groups={n_groups}."
            )
        group_dict = {}  # dict with groups and their indices
        for idx in range(n_samples):
            if groups[idx] not in group_dict:
                group_dict[groups[idx]] = [idx]
            else:
                group_dict[groups[idx]].append(idx)
        group_test_size = (
            self.test_size if self.test_size is not None else n_groups // n_folds
        )
        if n_groups - n_splits * group_test_size - self.gap <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of groups"
                f"={n_groups} with group_test_size={group_test_size} and gap={self.gap}."
            )

        group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)
        # loop over possible starts of the test set
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            train_end = group_test_start - gap
            if self.max_train_size and train_end > self.max_train_size:
                for train_group_idx in unique_groups[train_end - self.max_train_size: train_end]:
                    train_array_curr = group_dict[train_group_idx]  # indices of current train group
                    train_array = np.sort(np.concatenate((train_array, train_array_curr), axis=None), axis=None)
            else:
                for train_group_idx in unique_groups[:train_end]:
                    train_array_curr = group_dict[train_group_idx]  # indices of current train group
                    train_array = np.sort(np.concatenate((train_array, train_array_curr), axis=None), axis=None)
            for test_group_idx in unique_groups[group_test_start: group_test_start + group_test_size]:
                test_array_curr = group_dict[test_group_idx]  # indices of current test group
                test_array = np.sort(np.concatenate((test_array, test_array_curr), axis=None), axis=None)

            yield [int(i) for i in train_array], [int(i) for i in test_array]


