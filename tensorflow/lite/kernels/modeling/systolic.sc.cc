// Created by Nicolas Agostini

#include "tensorflow/lite/kernels/modeling/systolic.sc.h"

namespace tflite_soc {

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void SystolicDut::SetupGemm(int lhs_width_, int depth_, int rhs_width_,
               LhsScalar const* lhs_data_, RhsScalar const* rhs_data_,
               DstScalar* out_data_) {

  lhs_width = lhs_width_;
  depth = depth_;
  rhs_width = rhs_width_;
  lhs_data = lhs_data_;
  rhs_data = rhs_data_;
  out_data = out_data_;
}

template <typename DstScalar>
void SystolicDut::Test(DstScalar* data) {}

inline void ReLu(int & v) {
  if (v < 0)
    v = 0;
}

void SystolicDut::Gemm() {
  while (1) {
    for (int i = 0; i < lhs_width*rhs_width; i++)
      out_data[i]=0;

    for (int iA = 0; iA < lhs_width; iA++) {
      int tj = 0.0;
      for (int jA = 0; jA < depth; jA++) {
        int pA2 = iA * depth + jA;
        int tk = 0.0;
        for (int kB = 0; kB < rhs_width; kB++) {
          int pB2 = jA * rhs_width + kB;
          tk += rhs_data[pB2];
        }
        tj += lhs_data[pA2] * tk;
      }
      out_data[iA] = tj;
    }

    // Apply fused ReLu
    for (int i = 0; i < lhs_width*rhs_width; i++)
      ReLu(out_data[i]);
    wait();
  }
}

template void SystolicDut::Test<int>(int*);
template void SystolicDut::SetupGemm<int, int, int>(int, int, int, int const*,
                                                    int const*, int*);
}