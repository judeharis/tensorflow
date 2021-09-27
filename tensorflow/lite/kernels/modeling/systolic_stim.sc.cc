// Created by Nicolas Agostini
#include "tensorflow/lite/kernels/modeling/systolic_stim.sc.h"

namespace tflite_soc {

void SystolicStim::Method() {}

SystolicStim::SystolicStim(sc_module_name name_, bool debug_)
    : sc_module(name_), debug(debug_) {

  if (debug) {
    std::cout << "Running constructor of " << name() << std::endl;
  }
}

}  // namespace tflite_soc