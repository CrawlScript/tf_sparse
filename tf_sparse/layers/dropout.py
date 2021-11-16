# coding=utf-8
import tensorflow as tf
from tf_sparse.sparse_matrix import SparseMatrix


class Dropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):

        if isinstance(inputs, SparseMatrix):
            output_value = super().call(inputs.value, training)
            return inputs.with_value(output_value)

        elif isinstance(inputs, tf.sparse.SparseTensor):
            output_values = super().call(inputs.values, training)
            return inputs.with_values(output_values)

        else:
            return super().call(inputs, training)