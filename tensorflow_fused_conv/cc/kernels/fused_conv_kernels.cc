/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <iostream>

#include "fused_conv.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T>
struct FusedConvFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int in_size, int filter_size,
                  int add_size, int out_size, const T* in, const T* filter,
                  const T* add, T* out) {
    for (int out_iter = 0; out_iter < out_size * out_size; ++out_iter) {
      int out_hi = out_iter / out_size;
      int out_wi = out_iter % out_size;
      int in_base_hi = out_hi;
      int in_base_wi = out_wi;
      int sum = 0;
      for (int filter_iter = 0; filter_iter < filter_size * filter_size;
           ++filter_iter) {
        int filter_hi = filter_iter / filter_size;
        int filter_wi = filter_iter % filter_size;
        sum =
            sum +
            filter[filter_iter] *
                in[(in_base_hi + filter_hi) * in_size + in_base_wi + filter_wi];
      }
      out[out_iter] = sum + add[out_iter];
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class FusedConvOp : public OpKernel {
 public:
  explicit FusedConvOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input, filter and add tensor seperately
    const Tensor& input_tensor = context->input(0);
    const Tensor& filter_tensor = context->input(1);
    const Tensor& add_tensor = context->input(2);

    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in input tensor"));

    OP_REQUIRES(context, filter_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in filter tensor"));

    OP_REQUIRES(context, add_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in add tensor"));

    OP_REQUIRES(context,
                input_tensor.dims() == 2 &&
                    input_tensor.dim_size(0) == input_tensor.dim_size(1),
                errors::InvalidArgument("Input tensor's dimension should be 2 "
                                        "and height should be equal to width"));

    OP_REQUIRES(context,
                filter_tensor.dims() == 2 &&
                    filter_tensor.dim_size(0) == filter_tensor.dim_size(1),
                errors::InvalidArgument("Filter tensor's dimension should be 2 "
                                        "and height should be equal to width"));

    OP_REQUIRES(context,
                add_tensor.dims() == 2 &&
                    add_tensor.dim_size(0) == add_tensor.dim_size(1),
                errors::InvalidArgument("Add tensor's dimension should be 2 "
                                        "and height should be equal to width"));

    const int output_height =
        input_tensor.dim_size(0) - filter_tensor.dim_size(0) + 1;
    const int output_width =
        input_tensor.dim_size(1) - filter_tensor.dim_size(1) + 1;
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({output_height, output_width}),
                                &output_tensor));

    // Do the computation.
    FusedConvFunctor<Device, T>()(context->eigen_device<Device>(),
             static_cast<int>(input_tensor.dim_size(0)),
             static_cast<int>(filter_tensor.dim_size(0)),
             static_cast<int>(add_tensor.dim_size(0)),
             static_cast<int>(output_tensor->dim_size(0)),
             input_tensor.flat<T>().data(), filter_tensor.flat<T>().data(),
             add_tensor.flat<T>().data(), output_tensor->flat<T>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("FusedConv").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedConvOp<CPUDevice, T>);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                            \
  extern template struct FusedConvFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("FusedConv").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedConvOp<GPUDevice, T>);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace tensorflow
