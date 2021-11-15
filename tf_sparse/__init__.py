# coding=utf-8

import warnings

warnings.simplefilter('always', DeprecationWarning)

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.enable_eager_execution()


from . import utils
from .sparse_matrix import SparseMatrix
from .sparse_ops import sparse_diag_matmul, diag_sparse_matmul, add, minimum, maximum, diags, eye


# from . import nn, utils, data, datasets, layers, sparse
# from tf_geometric.data.graph import Graph, BatchGraph
# from tf_geometric.sparse.sparse_matrix import SparseMatrix
# from tf_geometric.sparse.sparse_adj import SparseAdj