// Created by Nicolas Agostini
#include <random>
#include <systemc/systemc.h>
#include "tensorflow/lite/kernels/modeling/systolic.sc.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite_soc {

using namespace tflite::cpu_backend_gemm;

template <typename Scalar>
void MakeDeterministicPseudoRandomVector(int size,
                                         std::vector<Scalar>* vector) {
  // Intentionally create a new local random_engine in each invocation,
  // so pseudorandom values don't depend on invocation order.
  // Otherwise, test results would be affecting by e.g. filtering.
  std::default_random_engine random_engine;
  (void)random_engine();
  // Do not use std::uniform*_distribution: the values that it
  // generates are implementation-defined.
  const double random_min = static_cast<double>(random_engine.min());
  const double random_max = static_cast<double>(random_engine.max());
  const double result_min =
      std::is_floating_point<Scalar>::value
          ? -1.0
          : std::max(-256., static_cast<double>(
                                std::numeric_limits<Scalar>::lowest()));
  const double result_max =
      std::is_floating_point<Scalar>::value
          ? 1.0
          : std::min(256.,
                     static_cast<double>(std::numeric_limits<Scalar>::max()));
  const double random_scale =
      (result_max - result_min) / (random_max - random_min);

  vector->resize(size);
  for (int i = 0; i < size; i++) {
    double val = random_scale * (random_engine() - random_min);
    val = std::max(val,
                   static_cast<double>(std::numeric_limits<Scalar>::lowest()));
    val =
        std::min(val, static_cast<double>(std::numeric_limits<Scalar>::max()));
    (*vector)[i] = static_cast<Scalar>(val);
  }
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
void TestGemm(int rows, int depth, int cols, SystolicDut& dut,
              sc_signal<bool>& trigger) {
  std::default_random_engine random_engine;
  std::vector<LhsScalar> lhs_data;
  std::vector<RhsScalar> rhs_data;
  std::vector<AccumScalar> bias_data;
  std::vector<DstScalar> dst_data;
  MakeDeterministicPseudoRandomVector(rows * depth, &lhs_data);
  MakeDeterministicPseudoRandomVector(depth * cols, &rhs_data);
  MakeDeterministicPseudoRandomVector(rows, &bias_data);

  MakeDeterministicPseudoRandomVector(rows * cols, &dst_data);
  
  MatrixParams<LhsScalar> lhs_params;
  lhs_params.order = Order::kRowMajor;
  lhs_params.rows = rows;
  lhs_params.cols = depth;
  if (!std::is_floating_point<LhsScalar>::value) {
    lhs_params.zero_point = 1;
    lhs_params.zero_point += random_engine() % 8;
  }

  MatrixParams<RhsScalar> rhs_params;
  rhs_params.order = Order::kColMajor;
  rhs_params.rows = depth;
  rhs_params.cols = cols;
  if (!std::is_floating_point<RhsScalar>::value) {
    rhs_params.zero_point = 1;
    rhs_params.zero_point += random_engine() % 8;
  }

  MatrixParams<DstScalar> dst_params;
  dst_params.order = Order::kColMajor;
  dst_params.rows = rows;
  dst_params.cols = cols;
  if (!std::is_floating_point<DstScalar>::value) {
    dst_params.zero_point = 1;
    dst_params.zero_point += random_engine() % 8;
  }

  GemmParams<AccumScalar, DstScalar> params;

  dut.SetupGemm(rows, depth, cols, 
                lhs_data.data(), 
                rhs_data.data(), 
                dst_data.data());

  dut.Test(dst_data.data());
  trigger = 1;
  sc_start(19,SC_NS);
  trigger = 0;
  sc_start(1,SC_NS);
  
  trigger = 1;
  sc_start(19,SC_NS);
  trigger = 0;
  sc_start(1,SC_NS);
}

}  // namespace tflite_soc

int sc_main(int argc, char* argv[]) {
  sc_clock clk_slow("ClkSlow", 100.0, SC_NS);
  sc_clock clk_fast("ClkFast", 50.0, SC_NS);

  sc_signal<bool> trigger;

  tflite_soc::SystolicDut s1("DUT", 64, true);
  s1.clock(clk_slow);
  s1.run_gemm(trigger);

  const int rows = 128;
  const int cols = 128;
  const int depth = 512;
  tflite_soc::TestGemm<int, int, int, int>(rows, depth, cols, s1, trigger);

  return (0);
}