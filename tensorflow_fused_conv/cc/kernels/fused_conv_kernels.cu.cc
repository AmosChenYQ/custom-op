/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>

#include "fused_conv.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// safe division
#define SDIV(x, y) (((x) + (y)-1) / (y))

#define H2D (cudaMemcpyHostToDevice)
#define D2H (cudaMemcpyDeviceToHost)
#define H2H (cudaMemcpyHostToHost)
#define D2D (cudaMemcpyDeviceToDevice)

// Define the CUDA kernel.
template <typename T>
__global__ void FusedConvCudaKernel(int in_size, int filter_size, int add_size,
                                    int out_size, const T* in, const T* filter,
                                    const T* add, T* out) {
  const int thid = blockDim.x * blockIdx.x + threadIdx.x;

  const int out_iter = thid;
  if (out_iter < out_size * out_size) {
    int out_hi = out_iter / out_size;
    int out_wi = out_iter % out_size;
    int in_base_hi = out_hi;
    int in_base_wi = out_wi;
    int sum = 0;
    for (int filter_iter = 0; filter_iter < filter_size * filter_size;
         ++filter_iter) {
      int filter_hi = filter_iter / filter_size;
      int filter_wi = filter_iter % filter_size;
      sum = sum +
            filter[filter_iter] *
                in[(in_base_hi + filter_hi) * in_size + in_base_wi + filter_wi];
    }

    out[out_iter] = sum + add[out_iter];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct FusedConvFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int in_size, int filter_size,
                  int add_size, int out_size, const T* in, const T* filter,
                  const T* add, T* out) {
    // Launch the cuda kernel to compute fusion of conv, add and relu kernel
    const int thread_per_block = 32;
    const int block_count = SDIV(out_size * out_size, thread_per_block);

    std::cout << "FusedConvFunctor block count " << block_count
              << " thread per block " << thread_per_block << std::endl;

    T *In = nullptr, *Filter = nullptr, *Add = nullptr, *Out = nullptr;
    cudaMalloc(&In, sizeof(T) * in_size * in_size);
    cudaMalloc(&Filter, sizeof(T) * filter_size * filter_size);
    cudaMalloc(&Add, sizeof(T) * add_size * add_size);
    cudaMalloc(&Out, sizeof(T) * out_size * out_size);

    cudaMemcpy(In, in, sizeof(T) * in_size * in_size, H2D);
    cudaMemcpy(Filter, filter, sizeof(T) * filter_size * filter_size, H2D);
    cudaMemcpy(Add, add, sizeof(T) * add_size * add_size, H2D);

    FusedConvCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(
        in_size, filter_size, add_size, out_size, In, Filter, Add, Out);

    cudaMemcpy(out, Out, sizeof(T) * out_size * out_size, D2H);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct FusedConvFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
