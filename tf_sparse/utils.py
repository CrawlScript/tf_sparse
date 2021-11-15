# coding=utf-8
import tensorflow as tf


def convert_sparse_index_to_hash(index, hash_key=None):
    edge_index_is_tensor = tf.is_tensor(index)
    num_nodes_is_none = hash_key is None
    num_nodes_is_tensor = tf.is_tensor(hash_key)

    index = tf.cast(index, tf.int64)

    # if not edge_index_is_tensor or edge_index.dtype != tf.int64:
    #     edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int64)

    if num_nodes_is_none:
        hash_key = tf.reduce_max(index) + 1
    else:
        hash_key = tf.cast(hash_key, tf.int64)

    row, col = index[0], index[1]

    hash = hash_key * row + col

    if tf.executing_eagerly() and not edge_index_is_tensor:
        hash = hash.numpy()

    if num_nodes_is_none:
        if tf.executing_eagerly() and not edge_index_is_tensor:
            hash_key = hash_key.numpy()
    else:
        if tf.executing_eagerly() and not num_nodes_is_tensor:
            hash_key = hash_key.numpy()

    return hash, hash_key


def convert_hash_to_sparse_index(hash, hash_key):
    edge_hash_is_tensor = tf.is_tensor(hash)

    # if not edge_hash_is_tensor:
    #     edge_hash = tf.convert_to_tensor(edge_hash)

    hash = tf.cast(hash, tf.int64)
    hash_key = tf.cast(hash_key, tf.int64)

    row = tf.math.floordiv(hash, hash_key)
    col = tf.math.floormod(hash, hash_key)

    edge_index = tf.stack([row, col], axis=0)
    edge_index = tf.cast(edge_index, tf.int32)

    if tf.executing_eagerly() and not edge_hash_is_tensor:
        edge_index = edge_index.numpy()

    return edge_index


def merge_duplicated_sparse_index(sparse_index, props=None, merge_modes=None):
    """
    merge_modes: list of merge_mode ("min", "max", "mean", "sum")
    """

    if props is not None:
        if type(merge_modes) is not list:
            raise Exception("type error: merge_modes should be a list of strings")
        if merge_modes is None:
            merge_modes = ["sum"] * len(props)

    # if edge_props is not None and merge_modes is None:
    #     raise Exception("merge_modes is required if edge_props is provided")

    sparse_index_is_tensor = tf.is_tensor(sparse_index)
    # edge_props_is_tensor = [tf.is_tensor(edge_prop) for edge_prop in edge_props]

    if not sparse_index_is_tensor:
        sparse_index = tf.convert_to_tensor(sparse_index, dtype=tf.int32)

    # if edge_props is not None:
    #     edge_props = [
    #         tf.convert_to_tensor(edge_prop) if edge_prop is not None else None
    #         for edge_prop in edge_props
    #     ]

    hash, hash_key = convert_sparse_index_to_hash(sparse_index)
    unique_hash, unique_index = tf.unique(hash)

    unique_sparse_index = convert_hash_to_sparse_index(unique_hash, hash_key)

    if tf.executing_eagerly() and not sparse_index_is_tensor:
        unique_sparse_index = unique_sparse_index.numpy()

    if props is None:
        unique_props = None
    else:
        unique_props = []
        for prop, merge_mode in zip(props, merge_modes):

            if prop is None:
                unique_prop = None
            else:

                prop_is_tensor = tf.is_tensor(prop)
                prop = tf.convert_to_tensor(prop)

                if merge_mode == "min":
                    merge_func = tf.math.unsorted_segment_min
                elif merge_mode == "max":
                    merge_func = tf.math.unsorted_segment_max
                elif merge_mode == "mean":
                    merge_func = tf.math.unsorted_segment_mean
                elif merge_mode == "sum":
                    merge_func = tf.math.unsorted_segment_sum
                else:
                    raise Exception("wrong merge mode: {}".format(merge_mode))
                unique_prop = merge_func(prop, unique_index, tf.shape(unique_hash)[0])

                if tf.executing_eagerly() and not prop_is_tensor:
                    unique_prop = unique_prop.numpy()

            unique_props.append(unique_prop)

    return unique_sparse_index, unique_props


# def convert_sparse_index_to_upper(sparse_index, props=None, merge_modes=None):
#     """
#
#     :param sparse_index:
#     :param props:
#     :param merge_modes: List of merge modes. Merge Modes: "min" | "max" | "mean" | "sum"
#     :return:
#     """
#
#     edge_index_is_tensor = tf.is_tensor(sparse_index)
#
#     if not edge_index_is_tensor:
#         sparse_index = tf.convert_to_tensor(sparse_index, dtype=tf.int32)
#
#     row = tf.math.reduce_min(sparse_index, axis=0)
#     col = tf.math.reduce_max(sparse_index, axis=0)
#
#     upper_edge_index = tf.stack([row, col], axis=0)
#     upper_edge_index, upper_edge_props = merge_duplicated_sparse_index(upper_edge_index, props, merge_modes)
#
#     if not edge_index_is_tensor:
#         upper_edge_index = upper_edge_index.numpy()
#
#     return upper_edge_index, upper_edge_props

