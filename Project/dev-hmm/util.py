# Copyright 2013-2014 Lars Buitinck / University of Amsterdam
#
# Some Parts Modified from scikit-learn, written by

import numpy as np
import scipy.sparse as sp
from sklearn.utils import check_random_state
from scipy.sparse import isspmatrix_csc, isspmatrix_csr

from scipy.sparse import csr_matrix
from sklearn.externals import six


def _assert_all_finite(X):
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # there everything is finite; fall back to O(n) space np.isfinite to
    # prevent false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


def assert_all_finite(X):
    """Throw a ValueError if X contains NaN or infinity.

    Input MUST be an np.ndarray instance or a scipy.sparse matrix."""

    _assert_all_finite(X.data if sp.issparse(X) else X)


def array2d(X, dtype=None, order=None, copy=False):
    if sp.issparse(X):
        raise TypeError('A sparse matrix was passed, but dense data '
                        'is required. Use X.toarray() to convert to dense.')
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = safe_copy(X_2d)
    return X_2d


def _atleast2d_or_sparse(X, dtype, order, copy, sparse_class, convmethod,
                         check_same_type):
    if sp.issparse(X):
        if check_same_type(X) and X.dtype == dtype:
            X = getattr(X, convmethod)(copy=copy)
        elif dtype is None or X.dtype == dtype:
            X = getattr(X, convmethod)()
        else:
            X = sparse_class(X, dtype=dtype)
        _assert_all_finite(X.data)
        X.data = np.array(X.data, copy=False, order=order)
    else:
        X = array2d(X, dtype=dtype, order=order, copy=copy)
    return X


def atleast2d_or_csr(X, dtype=None, order=None, copy=False):
    return _atleast2d_or_sparse(X, dtype, order, copy, sp.csr_matrix,
                                "tocsr", sp.isspmatrix_csr)


def validate_lengths(n_samples, lengths):
    if lengths is None:
        lengths = [n_samples]
    lengths = np.asarray(lengths, dtype=np.int32)
    if lengths.sum() > n_samples:
        msg = "More than {0:d} samples in lengths array {1!s}"
        raise ValueError(msg.format(n_samples, lengths))

    end = np.cumsum(lengths)
    start = end - lengths

    return start, end

def make_trans_matrix(y, n_classes, dtype=np.float64):
    indices = np.empty(len(y), dtype=np.int32)

    for i in six.moves.xrange(len(y) - 1):
        indices[i] = y[i] * i + y[i + 1]

    indptr = np.arange(len(y) + 1)
    indptr[-1] = indptr[-2]

    return csr_matrix((np.ones(len(y), dtype=dtype), indices, indptr),
                      shape=(len(y), n_classes ** 2))

def count_trans(y, n_classes):
    trans = np.zeros((n_classes, n_classes), dtype=np.intp)

    for i in range(y.shape[0] - 1):
        trans[y[i], y[i + 1]] += 1
    return trans

def safe_add(A, B):
    if isinstance(B, np.ndarray):
        A += B
        return

    if isspmatrix_csc(B):
        A = A.T
    elif not isspmatrix_csr(B):
        raise TypeError("Type {0} not supported.".format(type(B)))

    data = B.data
    indices = B.indices
    indptr = B.indptr

    for i in range(A.shape[0]):
        for jj in range(indptr[i], indptr[i + 1]):
            j = indices[jj]
            A[i, j] += data[jj]
