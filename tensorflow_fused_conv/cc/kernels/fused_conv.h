// kernel_example.h
#ifndef KERNEL_FUSED_CONV_H_
#define KERNEL_FUSED_CONV_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct FusedConvFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_FUSED_CONV_H_
