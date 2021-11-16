# coding=utf-8

import tensorflow as tf
from .sparse_matrix import SparseMatrix


def shape(x, out_type=tf.int32):
    if isinstance(x, SparseMatrix):
        x_shape = x.shape_tensor
        x_shape = tf.cast(x_shape, out_type)
    else:
        x_shape = tf.shape(x, out_type=out_type)
    return x_shape


def concat(items, axis):
    if len(items) == 0:
        return items
    sparse_tensors = [item.to_sparse_tensor() for item in items]
    output_sparse_tensor = tf.sparse.concat(axis, sparse_tensors)
    sparse_matrix_class = items[0].__class__
    return sparse_matrix_class.from_sparse_tensor(output_sparse_tensor)


# sparse_adj @ diagonal_matrix
def sparse_diag_matmul(sparse: SparseMatrix, diagonal):
    return sparse.matmul_diag(diagonal)


# self @ diagonal_matrix
def diag_sparse_matmul(diagonal, sparse: SparseMatrix):
    return sparse.rmatmul_diag(diagonal)


# element-wise sparse_adj addition
def add(a: SparseMatrix, b: SparseMatrix):
    return a + b


# element-wise sparse_adj subtraction
def subtract(a: SparseMatrix, b: SparseMatrix):
    return a - b


# element-wise maximum(a, b)
def maximum(a: SparseMatrix, b: SparseMatrix):
    return a.merge(b, merge_mode="max")


# element-wise minimum(a, b)
def minimum(a: SparseMatrix, b: SparseMatrix):
    return a.merge(b, merge_mode="min")


# Construct a SparseAdj from diagonals
def diags(diagonals):
    return SparseMatrix.from_diagonals(diagonals)


# Construct a SparseAdj with ones on diagonal
def eye(num_nodes):
    return SparseMatrix.eye(num_nodes)