// Created by Nicolas Agostini

#ifndef TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_H_

#include <systemc/systemc.h>
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite_soc {

SC_MODULE(SystolicDut) {

  sc_in<bool> clock;
  sc_in<bool> run_gemm;

  //  So in the LHS MxK matrix, the depth is K and the width in M.
  // And in the RHS KxN matrix, the depth is K and the width in N.
  //
  // This is illustrated in this picture:
  //
  //                             RHS width
  //                        <----------------->
  //                        +-----------------+ ^
  //                        |       RHS       | | Depth
  //                        +-----------------+ v
  //                 ^ +--+ +-----------------+
  //                 | |L | |                 |
  //       LHS width | |H | |      Result     |
  //                 | |S | |                 |
  //                 v +--+ +-----------------+
  //                   <-->
  //                   Depth
  void Gemm();

  // Function to test template
  template <typename DstScalar>
  void Test(DstScalar * data);

  template <typename LhsScalar, typename RhsScalar, typename DstScalar>
  void SetupGemm(int lhs_width_, int depth_, int rhs_width_,
                 LhsScalar const* lhs_data_, RhsScalar const* rhs_data_,
                 DstScalar* out_data_);

  SC_HAS_PROCESS(SystolicDut);

  // Parameters for the DUT
  SystolicDut(sc_module_name name_, int size_=64, bool debug_ = false) :
    sc_module(name_), size(size_), debug(debug_)
  {
    SC_THREAD(Gemm);
    sensitive << run_gemm.pos();

    buffer = new int[size];
    if (debug) {
      cout << "Running constructor of " << name() << endl;
    }
  }

  private:
    int * buffer;
    const int size;
    const bool debug;

    int lhs_width;
    int depth;
    int rhs_width;
    int const* lhs_data;
    int const* rhs_data;
    int* out_data;
};

// template<> void SystolicDut::Test<int>( int*);
}
#endif