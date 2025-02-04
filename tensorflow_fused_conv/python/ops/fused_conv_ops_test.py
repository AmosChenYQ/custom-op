# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for fused_conv ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from tensorflow_fused_conv.python.ops import fused_conv_ops
except ImportError:
  print("Cann't import fused_conv_ops from already installed packages")
  import fused_conv_ops


class FusedConvTest(test.TestCase):

  @test_util.run_gpu_only
  def testFusedConvGPU(self):
    print("test on gpu")
    with self.cached_session():
      with ops.device("/gpu:0"):
        self.assertAllClose(
          fused_conv_ops.fused_conv([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 1], [1, 1]], [[1, 2], [3, 4]]), np.array([[13, 18], [27, 32]]))

  def testFusedConvCPU(self):
    print("test on cpu")
    with self.cached_session():
      with ops.device("/cpu:0"):
        self.assertAllClose(
          fused_conv_ops.fused_conv([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 1], [1, 1]], [[1, 2], [3, 4]]), np.array([[13, 18], [27, 32]]))


if __name__ == '__main__':
  test.main()
