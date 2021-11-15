# coding=utf-8

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.linalg.sparse.sparse_csr_matrix_ops import sparse_tensor_to_csr_sparse_matrix, \
    sparse_matrix_sparse_mat_mul, csr_sparse_matrix_to_sparse_tensor
import warnings

from tf_sparse.utils import merge_duplicated_sparse_index

"""
Sparse Matrix for Computation
"""


def _segment_softmax(data, segment_ids, num_segments):
    max_values = tf.math.unsorted_segment_max(data, segment_ids, num_segments=num_segments)
    gathered_max_values = tf.gather(max_values, segment_ids)
    exp = tf.exp(data - tf.stop_gradient(gathered_max_values))
    denominator = tf.math.unsorted_segment_sum(exp, segment_ids, num_segments=num_segments) + 1e-8
    gathered_denominator = tf.gather(denominator, segment_ids)
    score = exp / gathered_denominator
    return score


class SparseMatrix(object):
    # https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    __array_priority__ = 10000

    def __init__(self, index, value=None, shape=None, merge=False):
        """
        Sparse Matrix for efficient computation.
        :param index:
        :param value:
        :param shape: [num_rows, num_cols], shape of the adjacency matrix.
        :param merge: Whether to merge duplicated edge
        """

        self.index = SparseMatrix.cast_index(index)

        index_is_tensor = tf.is_tensor(index)

        if value is not None:
            self.value = SparseMatrix.cast_value(value)
        else:
            if index_is_tensor:
                num_values = tf.shape(self.index)[1]
                value = tf.ones([num_values], dtype=tf.float32)
            else:
                num_values = np.shape(self.index)[1]
                value = np.ones([num_values], dtype=np.float32)
            self.value = value

        if merge:
            self.index, [self.value] = merge_duplicated_sparse_index(self.index, [self.value],
                                                                     merge_modes=["sum"])

        if shape is None:
            if index_is_tensor:
                num_nodes = tf.reduce_max(index) + 1
            else:
                num_nodes = np.max(index) + 1
            self.shape = [num_nodes, num_nodes]
        else:
            self.shape = shape

    @classmethod
    def cast_index(cls, index):
        index = tf.convert_to_tensor(index)
        index = tf.cast(index, tf.int32)
        return index
        # if isinstance(index, list):
        #     index = np.array(index).astype(np.int32)
        # elif isinstance(index, np.ndarray):
        #     index = index.astype(np.int32)
        # elif tf.is_tensor(index):
        #     index = tf.cast(index, tf.int32)

    @classmethod
    def cast_value(cls, value):
        value = tf.convert_to_tensor(value)
        value = tf.cast(value, dtype=tf.float32)
        return value

        # if isinstance(value, list):
        #     value = np.array(value).astype(np.float32)
        # elif isinstance(value, np.ndarray):
        #     value = value.astype(np.float32)
        # elif tf.is_tensor(value):
        #     value = tf.cast(value, tf.float32)
        # return value

    @property
    def row(self):
        return self.index[0]

    @property
    def col(self):
        return self.index[1]

    def merge_duplicated_index(self):
        edge_index, [edge_weight] = merge_duplicated_sparse_index(self.index, [self.value], merge_modes=["sum"])
        return self.__class__(edge_index, edge_weight, shape=self.shape)

    def negative(self):
        return self.__class__(
            index=self.index,
            value=-self.value,
            shape=self.shape
        )

    def __neg__(self):
        return self.negative()

    def transpose(self):
        row, col = self.index[0], self.index[1]
        transposed_edge_index = tf.stack([col, row], axis=0)
        return self.__class__(transposed_edge_index, value=self.value, shape=[self.shape[1], self.shape[0]])

    def map_value(self, map_func):
        return self.__class__(self.index, map_func(self.value), self.shape)

    def _reduce(self, segment_func, axis=-1, keepdims=False):

        # reduce by row
        if axis == -1 or axis == 1:
            reduce_axis = 0
        # reduce by col
        elif axis == 0 or axis == -2:
            reduce_axis = 1
        else:
            raise Exception("Invalid axis value: {}, axis shoud be -1, -2, 0, or 1".format(axis))

        reduce_index = self.index[reduce_axis]
        num_reduced = self.shape[reduce_axis]

        reduced_edge_weight = segment_func(self.value, reduce_index, num_reduced)
        if keepdims:
            reduced_edge_weight = tf.expand_dims(reduced_edge_weight, axis=axis)
        return reduced_edge_weight

    def reduce_sum(self, axis=-1, keepdims=False):
        return self._reduce(tf.math.unsorted_segment_sum, axis=axis, keepdims=keepdims)

    def reduce_min(self, axis=-1, keepdims=False):
        return self._reduce(tf.math.unsorted_segment_min, axis=axis, keepdims=keepdims)

    # https://stackoverflow.com/questions/45731484/tensorflow-how-to-perform-element-wise-multiplication-between-two-sparse-matrix
    def _mul_sparse_matrix(self, other):
        a, b = self, other
        ab = ((a + b).map_value(tf.square) - a.map_value(tf.square) - b.map_value(tf.square)).map_value(
            lambda x: x * 0.5)
        return ab

    def _rmul_sparse_matrix(self, other):
        return self._mul_sparse_matrix(other)

    def _mul_scalar(self, scalar):
        return self.__class__(self.index, self.value * scalar, self.shape)

    def _rmul_scalar(self, scalar):
        return self._mul_scalar(scalar)

    def _mul_dense(self, dense):
        sparse_tensor = self.to_sparse_tensor()
        return self.__class__.from_sparse_tensor(sparse_tensor * dense)

    def __mul__(self, other):

        if isinstance(other, SparseMatrix):
            return self._mul_sparse_matrix(other)
        # if is scalar
        elif len(tf.shape(other)) == 0:
            return self._mul_scalar(other)
        elif tf.is_tensor(other) or isinstance(other, np.ndarray):
            return self._mul_dense(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _matmul_dense(self, h):
        row, col = self.index[0], self.index[1]
        repeated_h = tf.gather(h, col)
        if self.value is not None:
            repeated_h *= tf.expand_dims(self.value, axis=-1)
        reduced_h = tf.math.unsorted_segment_sum(repeated_h, row, num_segments=self.shape[0])
        return reduced_h

    def _matmul_sparse(self, other):

        warnings.warn("The operation \"SparseMatrix @ SparseMatrix\" does not support gradient computation.")

        csr_matrix_a = self._to_csr_sparse_matrix()
        csr_matrix_b = other._to_csr_sparse_matrix()

        csr_matrix_c = sparse_matrix_sparse_mat_mul(
            a=csr_matrix_a, b=csr_matrix_b, type=self.value.dtype
        )

        sparse_tensor_c = csr_sparse_matrix_to_sparse_tensor(
            csr_matrix_c, type=self.value.dtype
        )

        output_class = self.__class__ if issubclass(self.__class__, other.__class__) else other.__class__

        return output_class.from_sparse_tensor(sparse_tensor_c)

    # sparse_adj @ other
    def matmul(self, other):
        if isinstance(other, SparseMatrix):
            return self._matmul_sparse(other)
        else:
            return self._matmul_dense(other)

    # h @ sparse_adj
    def rmatmul_dense(self, h):
        # h'
        transposed_h = tf.transpose(h, [1, 0])
        # sparse_adj' @ h'
        transpoed_output = self.transpose() @ transposed_h
        # h @ sparse_adj = (sparse_adj' @ h')'
        output = tf.transpose(transpoed_output, [1, 0])
        return output

    # # other_sparse_adj @ sparse_adj
    # def rmatmul_sparse(self, other):
    #     # h'
    #     transposed_other = other.transpose()
    #     # sparse_adj' @ h'
    #     transpoed_output = self.transpose() @ transposed_other
    #     # h @ sparse_adj = (sparse_adj' @ h')'
    #     output = transpoed_output.transpose()
    #     return output

    # self @ diagonal_matrix
    def matmul_diag(self, diagonal):
        col = self.index[1]
        updated_edge_weight = self.value * tf.gather(diagonal, col)
        return self.__class__(self.index, updated_edge_weight, self.shape)

    # self @ diagonal_matrix
    def rmatmul_diag(self, diagonal):
        row = self.index[0]
        updated_edge_weight = tf.gather(diagonal, row) * self.value
        return self.__class__(self.index, updated_edge_weight, self.shape)

    # self @ other (other is a dense tensor or SparseAdj)
    def __matmul__(self, other):
        return self.matmul(other)

    # h @ self (h is a dense tensor)
    def __rmatmul__(self, h):
        return self.rmatmul_dense(h)

    def eliminate_zeros(self):
        # edge_index_is_tensor = tf.is_tensor(self.index)
        # edge_weight_is_tensor = tf.is_tensor(self.value)

        mask = tf.not_equal(self.value, 0.0)
        masked_edge_index = tf.boolean_mask(self.index, mask, axis=1)
        masked_edge_weight = tf.boolean_mask(self.value, mask)

        # if not edge_index_is_tensor:
        #     masked_edge_index = masked_edge_index.numpy()
        #
        # if not edge_weight_is_tensor:
        #     masked_edge_weight = masked_edge_weight.numpy()

        return self.__class__(masked_edge_index, masked_edge_weight, shape=self.shape)

    def merge(self, other, merge_mode):
        """
        element-wise merge

        :param other:
        :param merge_mode:
        :return:
        """

        # edge_index_is_tensor = tf.is_tensor(self.index)
        # edge_weight_is_tensor = tf.is_tensor(self.value)

        combined_index = tf.concat([self.index, other.index], axis=1)
        combined_value = tf.concat([self.value, other.value], axis=0)

        merged_index, [merged_value] = merge_duplicated_sparse_index(
            combined_index, props=[combined_value], merge_modes=[merge_mode])

        # if tf.executing_eagerly() and not edge_index_is_tensor:
        #     merged_index = merged_index.numpy()
        #
        # if tf.executing_eagerly() and not edge_weight_is_tensor:
        #     merged_value = merged_value.numpy()

        output_class = self.__class__ if issubclass(self.__class__, other.__class__) else other.__class__
        merged_sparse_adj = output_class(merged_index, merged_value, shape=self.shape)

        return merged_sparse_adj

    def __add__(self, other_sparse_adj):
        """
        element-wise sparse adj addition: self + other_sparse_adj
        :param other_sparse_adj:
        :return:
        """
        return self.merge(other_sparse_adj, merge_mode="sum")

    # def __radd__(self, other_sparse_adj):
    #     """
    #     element-wise sparse adj addition: other_sparse_adj + self
    #     :param other_sparse_adj:
    #     :return:
    #     """
    #     return other_sparse_adj + self

    def __sub__(self, other):
        return self + other.negative()

    # def __rsub__(self, other):
    #     return other.__sub__(self)

    def dropout(self, drop_rate, training=False):
        if training and drop_rate > 0.0:
            edge_weight = tf.compat.v2.nn.dropout(self.value, drop_rate)
        else:
            edge_weight = self.value
        return self.__class__(self.index, value=edge_weight, shape=self.shape)

    def softmax(self, axis=-1):

        # reduce by row
        if axis == -1 or axis == 1:
            reduce_index = self.index[0]
            num_reduced = self.shape[0]
        # reduce by col
        elif axis == 0 or axis == -2:
            reduce_index = self.index[1]
            num_reduced = self.shape[1]
        else:
            raise Exception("Invalid axis value: {}, axis shoud be -1, -2, 0, or 1".format(axis))

        normed_value = _segment_softmax(self.value, reduce_index, num_reduced)

        return self.__class__(self.index, normed_value, shape=self.shape)

    # def add_self_loop(self, fill_weight=1.0):
    #     num_nodes = self.shape[0]
    #     updated_edge_index, updated_edge_weight = add_self_loop_edge(self.index, num_nodes,
    #                                                                  edge_weight=self.value,
    #                                                                  fill_weight=fill_weight)
    #     return SparseMatrix(updated_edge_index, updated_edge_weight, self.shape)
    #
    # def remove_self_loop(self):
    #     updated_edge_index, updated_edge_weight = remove_self_loop_edge(self.index, edge_weight=self.value)
    #     return SparseMatrix(updated_edge_index, updated_edge_weight, self.shape)

    def _to_csr_sparse_matrix(self):

        return sparse_tensor_to_csr_sparse_matrix(
            indices=tf.cast(tf.transpose(self.index, [1, 0]), tf.int64),
            values=self.value,
            dense_shape=self.shape
        )

    @classmethod
    def from_diagonals(cls, diagonals):
        """
        Construct a SparseAdj from diagonals
        :param diagonals:
        :return:
        """
        num_rows = tf.shape(diagonals)[0]
        row = tf.range(0, num_rows, dtype=tf.int32)
        col = row
        index = tf.stack([row, col], axis=0)
        return cls(index, diagonals, shape=[num_rows, num_rows])

    @classmethod
    def eye(cls, num_rows):
        """
        Construct a SparseAdj with ones on diagonal
        :param diagonals:
        :return:
        """
        diagonals = tf.ones([num_rows], dtype=tf.float32)
        return cls.from_diagonals(diagonals)

    @classmethod
    def from_sparse_tensor(cls, sparse_tensor: tf.sparse.SparseTensor):
        return cls(
            index=tf.transpose(sparse_tensor.indices, [1, 0]),
            value=sparse_tensor.values,
            shape=sparse_tensor.dense_shape
        )

    def to_sparse_tensor(self):

        sparse_tensor = tf.sparse.SparseTensor(
            indices=tf.cast(tf.transpose(self.index, [1, 0]), tf.int64),
            values=self.value,
            dense_shape=self.shape
        )
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        return sparse_tensor

    def to_dense(self):
        return tf.sparse.to_dense(self.to_sparse_tensor())

    def __str__(self):
        return "SparseMatrix: \n" \
               "index => \n" \
               "{}\n" \
               "value => {}\n" \
               "shape => {}".format(self.index, self.value, self.shape)

    def __repr__(self):
        return self.__str__()
