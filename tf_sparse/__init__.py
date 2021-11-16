# coding=utf-8

import warnings

warnings.simplefilter('always', DeprecationWarning)

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.enable_eager_execution()


from . import layers, utils
from .sparse_matrix import SparseMatrix, SparseMatrixSpec
from .sparse_ops import shape, concat, sparse_diag_matmul, diag_sparse_matmul, add, minimum, maximum, diags, eye

