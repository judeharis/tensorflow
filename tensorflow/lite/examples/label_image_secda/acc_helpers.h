#ifndef gd_HELPERS
#define gd_HELPERS

#include "tensorflow/lite/examples/label_image_secda/gemm_driver.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <typeinfo>
#include <cmath>
#include <fstream>



namespace tflite_secda {

template <typename Integer>
void Populate_Buff(gemm_driver &gd, int free, int inl0, int inl1, int inl2, int inl3){
  gd.dinb[free].inl0 = inl0;
  gd.dinb[free].inl1 = inl1;
  gd.dinb[free].inl2 = inl2;
  gd.dinb[free].inl3 = inl3;
  gd.dinb[free].in_use = true;
  gd.dinb[free].id = gd.dID++;
}

template <typename Integer>
int Check_For_Free_Buffer(gemm_driver &gd){
  for(int i=0; i<gd.bufflen;i++){
    if(!gd.dinb[i].in_use) return i;
  }
  return -1;
}

template <typename Integer>
int Find_Buff(gemm_driver &gd, int ID){
  for(int i=0; i<gd.bufflen;i++){
    if(gd.dinb[i].id == ID) return i;
  }
  return -1;
}

template <typename Integer>
bool Check_Done(gemm_driver &gd) {
  unsigned int s2mm_status = dma_get<int>(gd.dma0, S2MM_STATUS_REGISTER);
  bool done = (!(s2mm_status & 1<<12)) || (!(s2mm_status & 1<<1));
  return !done;
}

template <typename Integer>
int In_Use_Count(gemm_driver &gd){
  int count=0;
  for(int i=0; i<gd.bufflen;i++){
    if(gd.dinb[i].in_use) count++;
  }
  return count;
}
}
#endif