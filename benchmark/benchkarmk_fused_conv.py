import numpy as np
import tensorflow as tf
from tensorflow_fused_conv.python.ops import fused_conv_ops

input_matrix = np.ones((1, 1024, 1024, 1), dtype=np.int32)
filter_matrix = np.ones((2, 2, 1, 1), dtype=np.int32)
add_matrix = np.ones((1, 1023, 1023, 1), dtype=np.int32)

tf_conv_result = tf.nn.conv2d(input_matrix, filter_matrix, data_format="NHWC", strides=1, padding="VALID")
tf_fused_result = tf.add(tf_conv_result, add_matrix)
tf_final_result = tf.squeeze(tf_fused_result)

print(tf_final_result)

input_matrix = tf.squeeze(input_matrix)
filter_matrix = tf.squeeze(filter_matrix)
add_matrix = tf.squeeze(add_matrix)

fused_result_matrix = fused_conv_ops.fused_conv(input_matrix, filter_matrix, add_matrix)

print(fused_result_matrix)