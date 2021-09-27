// Created by Nicolas Agostini
#ifndef TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_STIM_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_STIM_H_

#include <systemc/systemc.h>

namespace tflite_soc {

SC_MODULE(SystolicStim) {
  // Signal Declarations =======================================================
  sc_signal<bool> clock;

  // Methods and Processes Declarations ========================================
  void Method();
  
  // Constructors ==============================================================

  // This line must be uncommented if Default Constructor is removed
  // SC_HAS_PROCESS(SystolicStim);

  // Custom constructur with parameters
  SystolicStim(sc_module_name name_, bool debug_ = false);

  // Default Constructor
  SC_CTOR(SystolicStim) : debug(false) {}

  // Private Variables =========================================================
  const bool debug;
};

}  // namespace tflite_soc

#endif  // TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_STIM_H_