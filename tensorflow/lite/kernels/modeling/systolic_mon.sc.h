// Created by Nicolas Agostini
#ifndef TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_MON_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_MON_H_

#include <systemc/systemc.h>

namespace tflite_soc {

SC_MODULE(SystolicMon) {
  // Signal Declarations =======================================================
  sc_in<bool> clock;

  // Methods and Processes Declarations ========================================
  void Method();
  
  // Constructors ==============================================================

  // This line must be uncommented if Default Constructor is removed
  // SC_HAS_PROCESS(SystolicMon);

  // Custom constructur with parameters
  SystolicMon(sc_module_name name_, bool debug_ = false);

  // Default Constructor
  SC_CTOR(SystolicMon) : debug(false) {
    SC_THREAD(Method);
    sensitive << clock.pos();
  }

  // Private Variables =========================================================
  const bool debug;
};

}  // namespace tflite_soc

#endif  // TENSORFLOW_LITE_KERNELS_MODELING_SYSTOLIC_MON_H_