import numpy as np
import tensorflow as tf
from tensorflow_fused_conv.python.ops import fused_conv_ops

tf_input_matrix = np.ones((1, 1024, 1024, 1), dtype=np.int32)
tf_filter_matrix = np.ones((2, 2, 1, 1), dtype=np.int32)
tf_add_matrix = np.ones((1, 1023, 1023, 1), dtype=np.int32)

tf_conv_result = tf.nn.conv2d(tf_input_matrix, tf_filter_matrix, data_format="NHWC", strides=1, padding="VALID")
tf_fused_result = tf.add(tf_conv_result, tf_add_matrix)
tf_final_result = tf.squeeze(tf_fused_result)

print(tf_final_result)

input_matrix = np.ones((1024, 1024), dtype=np.int32)
filter_matrix = np.ones((2, 2), dtype=np.int32)
add_matrix = np.ones((1023, 1023), dtype=np.int32)
fused_result = fused_conv_ops.fused_conv(input_matrix, filter_matrix, add_matrix)

print(fused_result)

tf.assert_equal(tf_final_result, fused_result)

def bench_tf():
  tf_conv_result = tf.nn.conv2d(tf_input_matrix, tf_filter_matrix, data_format="NHWC", strides=1, padding="VALID")
  tf_fused_result = tf.add(tf_conv_result, tf_add_matrix)
  tf_final_result = tf.squeeze(tf_fused_result)

def bench_fused():
  fused_result = fused_conv_ops.fused_conv(input_matrix, filter_matrix, add_matrix)

import time

tf_start_time = time.time()
for i in range(100):
  bench_tf()
tf_end_time = time.time()

fused_start_time = time.time()
for i in range(100):
  bench_fused()
fused_end_time = time.time()

print(tf_end_time - tf_start_time, fused_end_time - fused_start_time)