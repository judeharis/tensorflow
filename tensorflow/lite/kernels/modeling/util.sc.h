// Created by Nicolas Agostini
#ifndef TENSORFLOW_LITE_KERNELS_MODELING_UTIL_SC_H_
#define TENSORFLOW_LITE_KERNELS_MODELING_UTIL_SC_H_

// #include <systemc/systemc.h>
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"

namespace tflite_soc {

using namespace tflite::cpu_backend_gemm;

template <typename typeScalar>
void PrintMatrix(const MatrixParams<typeScalar>& matrix_params,
                 const typeScalar* matrix_data) {
#define MAX_COLS 32

  // if (&matrix_params && matrix_data) {
  //   printf("\nMatrix (%d,%d)\n", matrix_params.rows, matrix_params.cols);
  //   for (unsigned i = 0; i < matrix_params.rows; ++i) {
  //     printf("\n");
  //     for (unsigned j = 0; j < matrix_params.cols; ++j) {
  //       printf("%d,", matrix_data[i * matrix_params.cols + j]);
  //       if (j > MAX_COLS) {
  //         printf("...");
  //         break;
  //       }
  //     }
  //   }
  //   printf("\n ~~~~~~");
  //   printf("\n ~~~~~~");
  // }




  ofstream myfile;
  myfile.open ("out_7.txt");
  if (&matrix_params && matrix_data) {
    printf("\nMatrix (%d,%d)\n", matrix_params.rows, matrix_params.cols);


    for (unsigned i = 0; i < matrix_params.cols; i++) {
      printf("\n");
      myfile  << "\n";
      for (unsigned j = 0; j < matrix_params.rows; j++) {
        printf("%d,", matrix_data[j + matrix_params.rows * i]);
        myfile  << (int) matrix_data[j + matrix_params.rows * i]<< ",";
      }
    }




    // int index=0;
    // for (int c = 0; c < matrix_params.cols;c++) {
    //   printf("\n");
    //   myfile << endl;
    //   for (int r = 0; r <  matrix_params.rows;r++) {
    //     printf("%d,", matrix_data[index]);
    //     myfile << (int) matrix_data[index] << ",";
    //     index++;
    //   }
    // }

    myfile  <<  "\n~~~~~~~~~~~~~~~~~~~~~~~~~";
    myfile  <<  "\n~~~~~~~~~~~~~~~~~~~~~~~~~";
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  }


#undef MAX_COLS
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
void PrintMatrices(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data
) {
  #define MAX_COLS 32

  if (&lhs_params && lhs_data) {
    printf("\nlhs (%d,%d)\n", lhs_params.rows, lhs_params.cols);
    for (unsigned i = 0; i < lhs_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < lhs_params.cols; ++j) {
        printf("%d,", lhs_data[i * lhs_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  }

  if (&rhs_params && rhs_data) {
    printf("\nrhs (%d,%d)\n", rhs_params.rows, rhs_params.cols);
    for (unsigned i = 0; i < rhs_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < rhs_params.cols; ++j) {
        printf("%d,", rhs_data[i * rhs_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  }

  if (&dst_params && dst_data) {
    printf("\nout (%d,%d)\n", dst_params.rows, dst_params.cols);
    for (unsigned i = 0; i < dst_params.rows; ++i) {
      printf("\n");
      for (unsigned j = 0; j < dst_params.cols; ++j) {
        printf("%d,", dst_data[i * dst_params.cols + j]);
        if (j > MAX_COLS) {
          printf("...");
          break;
        }
      }
    }
    printf("\n ~~~~~~");
    printf("\n ~~~~~~");
  #undef MAX_COLS
  }
}



// template <typename Integer>
// void Prep_Inputs(int start, int end, int i_c, int width, int depth, const uint8_t* rhs_d,
//                  uint8_t* inb_0, uint8_t* inb_1, uint8_t* inb_2, uint8_t* inb_3) {

//     uint8x16_t tmp0;
//     uint8x16_t tmp1;
//     uint8x16_t tmp2;
//     uint8x16_t tmp3;

//     int w = ((width + 3) - ((width + 3) % 4));
//     int d = ((depth + 15) - ((depth + 15) % 16));
//     int dm= d-16;

//     int d2 = depth*2;
//     int d3 = depth*3;
//     int d4 = depth*4;

//     for(int i=start; i<end;i++){
//         int id = i*d4;
//         int i0= id;
//         int i1= id+ depth;
//         int i2= id+ d2;
//         int i3= id+ d3;

//         for(int j=0; j<dm;j+=16){
//             tmp0= vld1q_u8(rhs_d + i0 + j);
//             tmp1= vld1q_u8(rhs_d + i1 + j);
//             tmp2= vld1q_u8(rhs_d + i2 + j);
//             tmp3= vld1q_u8(rhs_d + i3 + j);

//             vst1q_u8(inb_0+i_c, tmp0);
//             vst1q_u8(inb_1+i_c, tmp1);
//             vst1q_u8(inb_2+i_c, tmp2);
//             vst1q_u8(inb_3+i_c, tmp3);
//             i_c+=16;
//         }


//         for(int j=dm; j<d;j++){
//             if (j<depth){
//                 unsigned char w0 = rhs_d[i0+j];
//                 unsigned char w1 = rhs_d[i1+j];
//                 unsigned char w2 = rhs_d[i2+j];
//                 unsigned char w3 = rhs_d[i3+j];

//                 inb_0[i_c]=w0;
//                 inb_1[i_c]=w1;
//                 inb_2[i_c]=w2;
//                 inb_3[i_c++]=w3;
//             }else{
//                 inb_0[i_c]=0;
//                 inb_1[i_c]=0;
//                 inb_2[i_c]=0;
//                 inb_3[i_c++]=0;
//             }
//         }
//     }
// }


// // Simple Hello World module
// SC_MODULE(hello_world){
//     SC_CTOR(hello_world){} 
//     void say_hello(){cout << "Hello World.\n";
//   }
// };

void say_hello();

// // Ram module
// // Extracted from here: https://www.doulos.com/knowhow/systemc/faq/#q2
// SC_MODULE(ram) {

//   sc_in<bool> clock;
//   sc_in<bool> RnW;   // ReadNotWrite
//   sc_in<int> address;
//   sc_inout<int> data;

//   void ram_proc();

//   SC_HAS_PROCESS(ram);

//   ram(sc_module_name name_, int size_=64, bool debug_ = false) :
//     sc_module(name_), size(size_), debug(debug_)
//   {
//     SC_THREAD(ram_proc);
//     sensitive << clock.pos();

//     buffer = new int[size];
//     if (debug) {
//       cout << "Running constructor of " << name() << endl;
//     }
//   }

//   private:
//     int * buffer;
//     const int size;
//     const bool debug;
// };
}
#endif // TENSORFLOW_LITE_KERNELS_MODELING_UTIL_SC_H_