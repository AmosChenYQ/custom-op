CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python3

ZERO_OUT_SRCS = $(wildcard tensorflow_zero_out/cc/kernels/*.cc) $(wildcard tensorflow_zero_out/cc/ops/*.cc)
TIME_TWO_SRCS = tensorflow_time_two/cc/kernels/time_two_kernels.cc $(wildcard tensorflow_time_two/cc/kernels/*.h) $(wildcard tensorflow_time_two/cc/ops/*.cc)
FUSED_CONV_SRCS = tensorflow_fused_conv/cc/kernels/fused_conv_kernels.cc $(wildcard tensorflow_fused_conv/cc/kernels/*.h) $(wildcard tensorflow_fused_conv/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++14
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = tensorflow_zero_out/python/ops/_zero_out_ops.so

TIME_TWO_GPU_ONLY_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.cu.o
TIME_TWO_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.so

FUSED_CONV_GPU_ONLY_TARGET_LIB = tensorflow_fused_conv/python/ops/_fused_conv_ops.cu.o
FUSED_CONV_TARGET_LIB = tensorflow_fused_conv/python/ops/_fused_conv_ops.so

# zero_out op for CPU
zero_out_op: $(ZERO_OUT_TARGET_LIB)

$(ZERO_OUT_TARGET_LIB): $(ZERO_OUT_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

zero_out_test: tensorflow_zero_out/python/ops/zero_out_ops_test.py tensorflow_zero_out/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_zero_out/python/ops/zero_out_ops_test.py

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts


# time_two op for GPU
time_two_gpu_only: $(TIME_TWO_GPU_ONLY_TARGET_LIB)

$(TIME_TWO_GPU_ONLY_TARGET_LIB): tensorflow_time_two/cc/kernels/time_two_kernels.cu.cc
	$(NVCC) -std=c++14 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

time_two_op: $(TIME_TWO_TARGET_LIB)
$(TIME_TWO_TARGET_LIB): $(TIME_TWO_SRCS) $(TIME_TWO_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

time_two_test: tensorflow_time_two/python/ops/time_two_ops_test.py tensorflow_time_two/python/ops/time_two_ops.py $(TIME_TWO_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_time_two/python/ops/time_two_ops_test.py

fused_conv_gpu_only: $(FUSED_CONV_GPU_ONLY_TARGET_LIB)

$(FUSED_CONV_GPU_ONLY_TARGET_LIB): tensorflow_fused_conv/cc/kernels/fused_conv_kernels.cu.cc
	$(NVCC) -std=c++14 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

fused_conv_op: $(FUSED_CONV_TARGET_LIB)
$(FUSED_CONV_TARGET_LIB): $(FUSED_CONV_SRCS) $(FUSED_CONV_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

fused_conv_test: tensorflow_fused_conv/python/ops/fused_conv_ops_test.py tensorflow_fused_conv/python/ops/fused_conv_ops.py $(FUSED_CONV_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_fused_conv/python/ops/fused_conv_ops_test.py

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(FUSED_CONV_GPU_ONLY_TARGET_LIB) $(TIME_TWO_TARGET_LIB) $(FUSED_CONV_TARGET_LIB)
