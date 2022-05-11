// kernel_example.h
#ifndef KERNEL_FUSED_CONV_H_
#define KERNEL_FUSED_CONV_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct FusedConvFunctor {
  FusedConvFunctor() {};
  // these sizes are dimensions of their corresponding matrix 
  void operator()(const Device& d, int in_size, int filter_size, int add_size,
                  int out_size, const T* in, const T* filter, const T* add,
                  T* out);
  int num_devices_;
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_FUSED_CONV_H_
