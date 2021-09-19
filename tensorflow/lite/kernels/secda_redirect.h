#ifndef SECDA_REDIRECT_H 
#define SECDA_REDIRECT_H

// #include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/examples/label_image_secda/gemm_driver.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv{

enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  // kMultithreadOptimized is a mixture of an Eigen-based kernel when threads
  // are available and kGenericOptimized when we must use only one thread.
  kMultithreadOptimized,
  // The kernel uses use CBLAS interface for matrix multiplication.
  // It's fast when an optimized CBLAS implementation is available (e.g. Apple
  // Accelerate Framework), and it's slow when falling back to naive
  // implementation.
  kCblasOptimized,
};

template <KernelType kernel_type>
TfLiteStatus Eval2(gemm_driver &gd,TfLiteContext* context, TfLiteNode* node);

TfLiteStatus BeforeEval(gemm_driver &gd ,TfLiteContext* context , TfLiteNode* node);

}
}
}
}

#endif