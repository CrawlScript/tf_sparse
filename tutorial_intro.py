# coding=utf-8
import os

# Enable GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tf_sparse as tfs
import tensorflow as tf
import numpy as np

# ==================================== SparseMatrix Creation ====================================

# create a SparseMatrix with index, value, and shape
x = tfs.SparseMatrix(
    index=[[0, 0, 1, 3],
           [1, 2, 2, 1]],
    value=[0.9, 0.8, 0.1, 0.2],
    shape=[4, 4]
)

print(x)

# create an identity SparseMatrix
print(tfs.eye(5))

# create a diagonal SparseMatrix
print(tfs.SparseMatrix.from_diagonals([0.1, 0.5, 0.2, 0.4]))

# ==================================== Operations ====================================


# SparseMatrix * Scalar
print(x * 5.0)
print(5.0 * x)

# row/column-level softmax (empty elements are not considered)
print(x.segment_softmax(axis=-1))

# Applying dropout operation on SparseMatrix
print(x.dropout(0.5, training=True))

# convert a SparseMatrix into a dense Tensor
tf_dense_x = x.to_dense()

# convert a SparseMatrix into tf.sparse.SparseTensor
tf_sparse_x = x.to_sparse_tensor()
print(tf_sparse_x)

# create a SparseMatrix with a tf.sparse.SparseTensor
print(tfs.SparseMatrix.from_sparse_tensor(tf_sparse_x))

# create a dense matrix
dense_a = np.random.randn(4, 4).astype(np.float32)

# SparseMatrix @ DenseMatrix (@ denotes matrix multiplication)
print(x @ dense_a)

# DenseMatrix @ SparseMatrix (@ denotes matrix multiplication)
print(dense_a.T @ x.transpose())

# create another SparseMatrix
y = tfs.SparseMatrix(
    index=[[1, 2, 2, 3],
           [0, 1, 3, 0]],
    value=[0.1, 0.4, 0.5, 0.0],
    shape=[4, 4]
)

print(y)

# Element-wise SparseMatrix addition
print(x + y)

# Element-wise SparseMatrix subtraction
print(x - y)
