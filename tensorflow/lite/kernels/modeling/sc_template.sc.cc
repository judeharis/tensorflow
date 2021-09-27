// Created by Nicolas Agostini
#include "tensorflow/lite/kernels/modeling/sc_template.sc.h"

namespace tflite_soc {

void ModuleName::Method() {}

ModuleName::ModuleName(sc_module_name name_, bool debug_)
    : sc_module(name_), debug(debug_) {
  SC_THREAD(Method);
  sensitive << clock.pos();

  if (debug) {
    std::cout << "Running constructor of " << name() << std::endl;
  }
}

}  // namespace tflite_soc