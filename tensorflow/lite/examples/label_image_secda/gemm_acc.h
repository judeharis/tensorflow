#ifndef DATA_HANDLER
#define DATA_HANDLER


#include "tensorflow/lite/kernels/internal/common.h"
#include <iostream>
#include <cstring>
#include <chrono>
#include <typeinfo>
#include <cmath>
#include <fstream>
#include <thread>

#include <arm_neon.h>
#include "tensorflow/lite/examples/label_image_secda/gemm_driver.h"
#include "tensorflow/lite/examples/label_image_secda/acc_helpers.h"

namespace tflite_secda {

struct uint8_params{
  uint8_t* data;
  const uint8_t* immutable_data;
  int order;
  int rows;
  int cols; 
  int depth;
  int zero_point;

  void Init(uint8_t* data_, int order_, int row_, int cols_, int zero_point_){
    data=data_;
    order=order_;
    rows=row_;
    cols=cols_;
    depth=0;
    zero_point=zero_point_;
  }

  void Init(const uint8_t* data_, int order_, int row_,int cols_, int depth_, int zero_point_){
    immutable_data=data_;
    order=order_;
    rows=row_;
    cols=cols_;
    depth=depth_;
    zero_point=zero_point_;
  }
};

template <typename Integer>
void saveData(gemm_driver &gd, bool inputs, int inl0, int inl1, int inl2, int inl3){
    ofstream wrin0;ofstream wrin1;ofstream wrin2;ofstream wrin3;
    string filename = "model"+ std::to_string(gd.t.layer)+ "_w" + std::to_string(gd.t.layer_weight_tile);
    if(inputs) filename = "model"+ std::to_string(gd.t.layer)+ "_w" + std::to_string(gd.t.layer_weight_tile) + "_i" + std::to_string(gd.t.layer_input_tile);
    wrin0.open ("aData/"+filename+"_0.txt");
    wrin1.open ("aData/"+filename+"_1.txt");      
    wrin2.open ("aData/"+filename+"_2.txt");      
    wrin3.open ("aData/"+filename+"_3.txt");   
    for(int i=0;i<inl0;i++){
      wrin0 << gd.in0[i] << "\n";
      if(i<inl1)wrin1 << gd.in1[i] << "\n";
      if(i<inl2)wrin2 << gd.in2[i] << "\n";
      if(i<inl3)wrin3 << gd.in3[i] << "\n";
    }
    wrin0.close(); wrin1.close(); wrin2.close();wrin3.close();
}


template <typename Integer>
void Load_RHS_Data(gemm_driver &gd, int start_col, int cols,int real_depth, int depth){
    int inl0=0;
    int inl1=0;
    int inl2=0;
    int inl3=0;

    int offdepth =  real_depth*gd.rhs_offset;
    int start_dex = (start_col/4);
    int* p_rhs_sums1 = reinterpret_cast<int*> (&gd.in_sum1[start_dex]);
    int* p_rhs_sums2 = reinterpret_cast<int*> (&gd.in_sum2[start_dex]);
    int* p_rhs_sums3 = reinterpret_cast<int*> (&gd.in_sum3[start_dex]);
    int* p_rhs_sums4 = reinterpret_cast<int*> (&gd.in_sum4[start_dex]);

    int roundedcols = ((cols + 3) - ((cols + 3) % 4));
    int in_sum_length = roundedcols/4;
    std::uint32_t h= 1;
    uint32_t l=in_sum_length;l=l<<16;l+=roundedcols*depth/4;
    gd.in0[inl0++]= h;
    gd.in0[inl0++]= 0;
    gd.in0[inl0++]= l;
    gd.in0[inl0++]= gd.rf;
    gd.in0[inl0++]= gd.ra;
    gd.in0[inl0++]= gd.re;

#ifndef ACC_NEON
    for (int c = 0; c < cols; c += 4) {
      for(int i=0;i<depth/4;i++){
        gd.in0[inl0++]= gd.inb_0[i+gd.in_id];
        gd.in1[inl1++]= gd.inb_1[i+gd.in_id];
        gd.in2[inl2++]= gd.inb_2[i+gd.in_id];
        gd.in3[inl3++]= gd.inb_3[i+gd.in_id];
      }
      gd.in_id+=depth/4;
    }
    for(int i=0;i<in_sum_length;i++){
      gd.in0[inl0++]= (p_rhs_sums1[i] + offdepth) * gd.lhs_offset;
      gd.in1[inl1++]= (p_rhs_sums2[i] + offdepth) * gd.lhs_offset;
      gd.in2[inl2++]= (p_rhs_sums3[i] + offdepth) * gd.lhs_offset;
      gd.in3[inl3++]= (p_rhs_sums4[i] + offdepth) * gd.lhs_offset;
    }
#else
    for (int c = 0; c < cols; c += 4) {
      unsigned int* inb0 = gd.inb_0;
      unsigned int* inb1 = gd.inb_1;
      unsigned int* inb2 = gd.inb_2;
      unsigned int* inb3 = gd.inb_3;
      for(int i=0;i<depth/4;i+=4){
          vst1q_u32(gd.in0+inl0, vld1q_u32(inb0 + i + gd.in_id));
          vst1q_u32(gd.in1+inl1, vld1q_u32(inb1 + i + gd.in_id));
          vst1q_u32(gd.in2+inl2, vld1q_u32(inb2 + i + gd.in_id));
          vst1q_u32(gd.in3+inl3, vld1q_u32(inb3 + i + gd.in_id));
          inl0+=4;  
          inl1+=4;
          inl2+=4;  
          inl3+=4;
      }
      gd.in_id+=depth/4;
    }
    for(int i=0;i<in_sum_length;i++){
      gd.in0[inl0++]= (p_rhs_sums1[i] + offdepth) * gd.lhs_offset;
      gd.in1[inl1++]= (p_rhs_sums2[i] + offdepth) * gd.lhs_offset;
      gd.in2[inl2++]= (p_rhs_sums3[i] + offdepth) * gd.lhs_offset;
      gd.in3[inl3++]= (p_rhs_sums4[i] + offdepth) * gd.lhs_offset;
    }
#endif

    dma_set<int>(gd.dma0, MM2S_LENGTH, inl0*4);
    dma_set<int>(gd.dma1, MM2S_LENGTH, inl1*4);
    dma_set<int>(gd.dma2, MM2S_LENGTH, inl2*4);
    dma_set<int>(gd.dma3, MM2S_LENGTH, inl3*4);
    dma_mm2s_sync<int>(gd.dma0);
    dma_mm2s_sync<int>(gd.dma1);
    dma_mm2s_sync<int>(gd.dma2);
    dma_mm2s_sync<int>(gd.dma3);
    gd.lhs_start=true;
    // if(gd.t.layer==gd.t.layer_print && gd.t.layer_weight_tile==gd.t.layer_ww)saveData<int>(gd,false,inl0,inl1,inl2,inl3);
}


template <typename Integer>
void Load_LHS_Data(gemm_driver &gd, int free_buf, uint8_t* results, int dcs, int start_row, int rows, int start_col, int cols,int depth, int rcols, int rrows) {
    int offset = gd.dinb[free_buf].offset;
    unsigned int* in0 = gd.in0+(offset/4);
    unsigned int* in1 = gd.in1+(offset/4);
    unsigned int* in2 = gd.in2+(offset/4);
    unsigned int* in3 = gd.in3+(offset/4);

    int inl0=0;
    int inl1=0;
    int inl2=0;
    int inl3=0;

    int w_dex=(gd.w_c/4); 
    int data_length = depth*rows;
    int wt_sums_len = rows/4;

    uint32_t h= 0;
    uint32_t count=rows;count=count<<16;count+=cols;
    uint32_t l =rows*depth/4;l=l<<16;l+=wt_sums_len;
    h+= depth;h=h<<8;
    h+= 0;h=h<<8;
    h+= 0;h=h<<1;
    h+= 1;h=h<<1;
    in0[inl0++]= h;
    in0[inl0++]= count;
    in0[inl0++]= l;

#ifndef ACC_NEON
    for(int i=0;i<data_length/16;i++){
      in0[inl0++]= gd.wb_0[w_dex+i];
      in1[inl1++]= gd.wb_1[w_dex+i];
      in2[inl2++]= gd.wb_2[w_dex+i];
      in3[inl3++]= gd.wb_3[w_dex+i];
    }
#else
    for(int i=0;i<data_length/16;i+=4){
      vst1q_u32(in0+inl0, vld1q_u32(gd.wb_0 + w_dex + i));
      vst1q_u32(in1+inl1, vld1q_u32(gd.wb_1 + w_dex + i));
      vst1q_u32(in2+inl2, vld1q_u32(gd.wb_2 + w_dex + i));
      vst1q_u32(in3+inl3, vld1q_u32(gd.wb_3 + w_dex + i));
      inl0+=4;
      inl1+=4;
      inl2+=4;  
      inl3+=4;
    }
#endif

    int b_c = start_row;
    int start_dex= gd.dt_sum_dex[gd.t.layer] + (start_row/4);
    int* p_lhs_sums1= reinterpret_cast<int*> (&gd.wt_sum1[start_dex]);
    int* p_lhs_sums2= reinterpret_cast<int*> (&gd.wt_sum2[start_dex]);
    int* p_lhs_sums3= reinterpret_cast<int*> (&gd.wt_sum3[start_dex]);
    int* p_lhs_sums4= reinterpret_cast<int*> (&gd.wt_sum4[start_dex]);
    for(int i=0;i<wt_sums_len;i++){
      in0[inl0++]= (p_lhs_sums1[i] * gd.rhs_offset) + gd.bias[b_c++];
      in1[inl1++]= (p_lhs_sums2[i] * gd.rhs_offset) + gd.bias[b_c++];
      in2[inl2++]= (p_lhs_sums3[i] * gd.rhs_offset) + gd.bias[b_c++];
      in3[inl3++]= (p_lhs_sums4[i] * gd.rhs_offset) + gd.bias[b_c++];
    }
    gd.w_c+=data_length/4;

    in0[inl0++]= 4294967295;
    uint8_t* res_pointer = results + start_row + start_col * dcs;
    gd.st_params[free_buf].dst=reinterpret_cast<int*>  (res_pointer);
    gd.st_params[free_buf].dcs = dcs;
    gd.st_params[free_buf].rows = rows;
    gd.st_params[free_buf].cols = cols;
    gd.st_params[free_buf].rcols = rcols;
    gd.st_params[free_buf].rrows = rrows;
    Populate_Buff<int>(gd,free_buf, inl0,inl1,inl2,inl3);
    // if(gd.t.layer==gd.t.layer_print && gd.t.layer_weight_tile==gd.t.layer_ww && gd.t.layer_input_tile==gd.t.layer_iw)saveData<int>(gd,true,inl0,inl1,inl2,inl3);
}

template <typename Integer>
void Start_Transfer(gemm_driver &gd) {
    int s_buf = Find_Buff<int>(gd,gd.sID);
    dma_change_start<int>(gd.dma0,gd.dma1,gd.dma2,gd.dma3,gd.dinb[s_buf].offset);
    dma_set<int>(gd.dma0, MM2S_LENGTH, gd.dinb[s_buf].inl0*4);
    dma_set<int>(gd.dma1, MM2S_LENGTH, gd.dinb[s_buf].inl1*4);
    dma_set<int>(gd.dma2, MM2S_LENGTH, gd.dinb[s_buf].inl2*4);
    dma_set<int>(gd.dma3, MM2S_LENGTH, gd.dinb[s_buf].inl3*4);
    gd.sID++;
}


template <typename Integer>
void End_Transfer(gemm_driver &gd) {
    dma_mm2s_sync<int>(gd.dma0);
    dma_mm2s_sync<int>(gd.dma1);
    dma_mm2s_sync<int>(gd.dma2);
    dma_mm2s_sync<int>(gd.dma3);
}

template <typename Integer>
void Set_Results(gemm_driver &gd) {
    int s_buf = Find_Buff<int>(gd,gd.sID);
    dma_change_end<int>(gd.dma0,gd.dma1,gd.dma2,gd.dma3,gd.dinb[s_buf].offset);
    dma_set<int>(gd.dma0, S2MM_LENGTH,200000);
    dma_set<int>(gd.dma1, S2MM_LENGTH,200000);
    dma_set<int>(gd.dma2, S2MM_LENGTH,200000);
    dma_set<int>(gd.dma3, S2MM_LENGTH,200000);
}

template <typename Integer>
void Recieve_Results(gemm_driver &gd) {
    dma_s2mm_sync<int>(gd.dma0);
    dma_s2mm_sync<int>(gd.dma1);
    dma_s2mm_sync<int>(gd.dma2);
    dma_s2mm_sync<int>(gd.dma3);
  }


#ifdef VM_ACC
template <typename Integer>
void Store_Results(gemm_driver &gd) {
    int r_buf = Find_Buff<int>(gd,gd.rID);
    int offset = gd.dinb[r_buf].offset;
    struct store_params sp =  gd.st_params[r_buf];
    int dcs = sp.dcs; 
    int rows = sp.rows; 
    int cols = sp.cols;  
    int rcols = sp.rcols; 
    int rrows = sp.rrows; 
    uint8_t* base= reinterpret_cast<uint8_t*>  (sp.dst);
    gd.dinb[r_buf].in_use=false;
    gd.dinb[r_buf].id=-1;
    gd.rID++;

    int* o0 = gd.out0+(offset/4);
    int* o1 = gd.out1+(offset/4);
    int* o2 = gd.out2+(offset/4);
    int* o3 = gd.out3+(offset/4);
    uint8_t* bo0= reinterpret_cast<uint8_t*> (o0);
    uint8_t* bo1= reinterpret_cast<uint8_t*> (o1);
    uint8_t* bo2= reinterpret_cast<uint8_t*> (o2);
    uint8_t* bo3= reinterpret_cast<uint8_t*> (o3);

    int out0=0;
    int out1=0;
    int out2=0;
    int out3=0;
    int dcols = rcols - (rcols%4);
    int rowsdiff = rows - rrows;
    int r16 = rrows - rrows%16;

#ifndef ACC_NEON
  for(int i=0; i<dcols;i+=4){
    for(int j=0; j<rrows;j++){
      base[(i + 0) * dcs + j]  =  bo0[out0++];
      base[(i + 1) * dcs + j]  =  bo1[out1++];
      base[(i + 2) * dcs + j]  =  bo2[out2++];
      base[(i + 3) * dcs + j]  =  bo3[out3++];
    }
    out0+= rowsdiff;
    out1+= rowsdiff;
    out2+= rowsdiff;
    out3+= rowsdiff;
  }
#else
  for(int i=0; i<dcols;i+=4){
      int di0 = i*dcs; 
      int di1 = (i+1)*dcs; 
      int di2 = (i+2)*dcs; 
      int di3 = (i+3)*dcs; 
      for(int j=0; j<r16;j+=16){
        vst1q_u8(base + di0 + j,vld1q_u8(bo0 + out0));
        vst1q_u8(base + di1 + j,vld1q_u8(bo1 + out1));
        vst1q_u8(base + di2 + j,vld1q_u8(bo2 + out2));
        vst1q_u8(base + di3 + j,vld1q_u8(bo3 + out3));
        out0+=16; 
        out1+=16; 
        out2+=16; 
        out3+=16; 
      }
      for(int j=r16; j<rrows;j++){
        base[di0 + j]  =  bo0[out0++];
        base[di1 + j]  =  bo1[out1++];
        base[di2 + j]  =  bo2[out2++];
        base[di3 + j]  =  bo3[out3++];
      }
      out0+= rowsdiff;
      out1+= rowsdiff;
      out2+= rowsdiff;
      out3+= rowsdiff;
  }
#endif

    if((rcols%4)==3){
      for(int j=0; j<rrows;j++){
          base[(dcols + 0) * dcs + j]  =  bo0[out0++];
          base[(dcols + 1) * dcs + j]  =  bo1[out1++];
          base[(dcols + 2) * dcs + j]  =  bo2[out2++];
      }
      out0+= rowsdiff;
      out1+= rowsdiff;
      out2+= rowsdiff;
    }else if((rcols%4)==2){
      for(int j=0; j<rrows;j++){
          base[(dcols + 0) * dcs + j]  =  bo0[out0++];
          base[(dcols + 1) * dcs + j]  =  bo1[out1++];
      }
      out0+= rowsdiff;
      out1+= rowsdiff;
    }else if((rcols%4)==1){
      for(int j=0; j<rrows;j++){
          base[(dcols + 0) * dcs + j]  =  bo0[out0++];
      }
      out0+= rowsdiff;
    }
}

#else

template <typename Integer>
void Store_Results(gemm_driver &gd) {
    int r_buf = Find_Buff<int>(gd,gd.rID);
    int offset = gd.dinb[r_buf].offset;
    struct store_params sp =  gd.st_params[r_buf];
    int dcs = sp.dcs; 
    int rows = sp.rows; 
    int cols = sp.cols;  
    int rcols = sp.rcols; 
    int rrows = sp.rrows; 
    uint8_t* base= reinterpret_cast<uint8_t*>  (sp.dst);
    gd.dinb[r_buf].in_use=false;
    gd.dinb[r_buf].id=-1;
    gd.rID++;
    
    int* o0 = gd.out0+(offset/4);
    int* o1 = gd.out1+(offset/4);
    int* o2 = gd.out2+(offset/4);
    int* o3 = gd.out3+(offset/4);
    uint8_t* bo0= reinterpret_cast<uint8_t*> (o0);
    uint8_t* bo1= reinterpret_cast<uint8_t*> (o1);
    uint8_t* bo2= reinterpret_cast<uint8_t*> (o2);
    uint8_t* bo3= reinterpret_cast<uint8_t*> (o3);

    int out0=0;
    int out1=0;
    int out2=0;
    int out3=0;
    int r16 = rows - rows%16;
    int rcolsr = rcols%16;
    int dcols = rcols - (rcolsr);

#ifndef ACC_NEON
    for(int i=0; i<dcols;i+=16){
        for(int j=0; j<r16;j+=16){
          for(int k=0; k<16;k++){
            base[(i + 0) * dcs + (j) + k]  =  bo0[out0++];
            base[(i + 1) * dcs + (j) + k]  =  bo1[out1++];
            base[(i + 2) * dcs + (j) + k]  =  bo2[out2++];
            base[(i + 3) * dcs + (j) + k]  =  bo3[out3++];
          }
          for(int k=0; k<16;k++){
            base[(i + 4) * dcs + (j) + k]  =  bo0[out0++];
            base[(i + 5) * dcs + (j) + k]  =  bo1[out1++];
            base[(i + 6) * dcs + (j) + k]  =  bo2[out2++];
            base[(i + 7) * dcs + (j) + k]  =  bo3[out3++];
          }
          for(int k=0; k<16;k++){
            base[(i + 8) * dcs + (j) + k]  =  bo0[out0++];
            base[(i + 9) * dcs + (j) + k]  =  bo1[out1++];
            base[(i + 10) * dcs + (j) + k]  =  bo2[out2++];
            base[(i + 11) * dcs + (j) + k]  =  bo3[out3++];
          }
          for(int k=0; k<16;k++){
            base[(i + 12) * dcs + (j) + k]  =  bo0[out0++];
            base[(i + 13) * dcs + (j) + k]  =  bo1[out1++];
            base[(i + 14) * dcs + (j) + k]  =  bo2[out2++];
            base[(i + 15) * dcs + (j) + k]  =  bo3[out3++];
          }
        }

        for(int j=r16; j<rows;j++){
          base[(i + 0) * dcs + j]  =  bo0[out0++];
          base[(i + 1) * dcs + j]  =  bo1[out1++];
          base[(i + 2) * dcs + j]  =  bo2[out2++];
          base[(i + 3) * dcs + j]  =  bo3[out3++];
        }
        for(int j=r16; j<rows;j++){
          base[(i + 4) * dcs + j]  =  bo0[out0++];
          base[(i + 5) * dcs + j]  =  bo1[out1++];
          base[(i + 6) * dcs + j]  =  bo2[out2++];
          base[(i + 7) * dcs + j]  =  bo3[out3++];
        }
        for(int j=r16; j<rows;j++){
          base[(i + 8) * dcs + j]  =  bo0[out0++];
          base[(i + 9) * dcs + j]  =  bo1[out1++];
          base[(i + 10) * dcs + j]  =  bo2[out2++];
          base[(i + 11) * dcs + j]  =  bo3[out3++];
        }
        for(int j=r16; j<rows;j++){
          base[(i + 12) * dcs + j]  =  bo0[out0++];
          base[(i + 13) * dcs + j]  =  bo1[out1++];
          base[(i + 14) * dcs + j]  =  bo2[out2++];
          base[(i + 15) * dcs + j]  =  bo3[out3++];
        }
    }
#else
      for(int i=0; i<dcols;i+=16){
          int di0 = i*dcs; 
          int di1 = (i+1)*dcs; 
          int di2 = (i+2)*dcs; 
          int di3 = (i+3)*dcs; 
          int di4 = (i+4)*dcs; 
          int di5 = (i+5)*dcs; 
          int di6 = (i+6)*dcs; 
          int di7 = (i+7)*dcs; 
          int di8 = (i+8)*dcs; 
          int di9 = (i+9)*dcs; 
          int di10 = (i+10)*dcs; 
          int di11 = (i+11)*dcs; 
          int di12 = (i+12)*dcs; 
          int di13 = (i+13)*dcs; 
          int di14 = (i+14)*dcs; 
          int di15 = (i+15)*dcs; 

          for(int j=0; j<r16;j+=16){
            vst1q_u8(base + di0 + j,vld1q_u8(bo0 + out0));
            vst1q_u8(base + di1 + j,vld1q_u8(bo1 + out1));
            vst1q_u8(base + di2 + j,vld1q_u8(bo2 + out2));
            vst1q_u8(base + di3 + j,vld1q_u8(bo3 + out3));
            vst1q_u8(base + di4 + j,vld1q_u8(bo0 + out0+16));
            vst1q_u8(base + di5 + j,vld1q_u8(bo1 + out1+16));
            vst1q_u8(base + di6 + j,vld1q_u8(bo2 + out2+16));
            vst1q_u8(base + di7 + j,vld1q_u8(bo3 + out3+16));
            vst1q_u8(base + di8 + j,vld1q_u8(bo0 + out0+32));
            vst1q_u8(base + di9 + j,vld1q_u8(bo1 + out1+32));
            vst1q_u8(base + di10 + j,vld1q_u8(bo2 + out2+32));
            vst1q_u8(base + di11 + j,vld1q_u8(bo3 + out3+32));
            vst1q_u8(base + di12 + j,vld1q_u8(bo0 + out0+48));
            vst1q_u8(base + di13 + j,vld1q_u8(bo1 + out1+48));
            vst1q_u8(base + di14 + j,vld1q_u8(bo2 + out2+48));
            vst1q_u8(base + di15 + j,vld1q_u8(bo3 + out3+48));
            out0+=64; 
            out1+=64; 
            out2+=64; 
            out3+=64; 
          }
          for(int j=r16; j<rows;j++){
            base[di0 + j]  =  bo0[out0++];
            base[di1 + j]  =  bo1[out1++];
            base[di2 + j]  =  bo2[out2++];
            base[di3 + j]  =  bo3[out3++];
          }
          for(int j=r16; j<rows;j++){
            base[di4 + j]  =  bo0[out0++];
            base[di5 + j]  =  bo1[out1++];
            base[di6 + j]  =  bo2[out2++];
            base[di7 + j]  =  bo3[out3++];
          }
          for(int j=r16; j<rows;j++){
            base[di8 + j]  =  bo0[out0++];
            base[di9 + j]  =  bo1[out1++];
            base[di10 + j]  =  bo2[out2++];
            base[di11 + j]  =  bo3[out3++];
          }
          for(int j=r16; j<rows;j++){
            base[di12 + j]  =  bo0[out0++];
            base[di13 + j]  =  bo1[out1++];
            base[di14 + j]  =  bo2[out2++];
            base[di15 + j]  =  bo3[out3++];
          }
      }
#endif

    for(int j=0; j<r16;j+=16) {
      for(int i=0;i<rcolsr;i++){
        uint8_t* bos;
        int outs;
        if(i%4==0) {bos = bo0;outs = out0;}
        if(i%4==1) {bos = bo1;outs = out1;}
        if(i%4==2) {bos = bo2;outs = out2;}
        if(i%4==3) {bos = bo3;outs = out3;}
        for(int k=0; k<16;k++)base[(dcols + i) * dcs + j +k]  =  bos[outs++];
        if(i%4==0)out0 = outs;
        if(i%4==1)out1 = outs;
        if(i%4==2)out2 = outs;
        if(i%4==3)out3 = outs;
      }
    }
    for(int i=0;i<rcolsr;i++){
      uint8_t* bos;
      int outs;
      if(i%4==0) {bos = bo0;outs = out0;}
      if(i%4==1) {bos = bo1;outs = out1;}
      if(i%4==2) {bos = bo2;outs = out2;}
      if(i%4==3) {bos = bo3;outs = out3;}
      for(int j=r16; j<rows;j++)base[(dcols + i) * dcs + j]  =  bos[outs++];
      if(i%4==0)out0 = outs;
      if(i%4==1)out1 = outs;
      if(i%4==2)out2 = outs;
      if(i%4==3)out3 = outs;
    }
}
#endif


template <typename Integer>
void DataHandleComputeL1(gemm_driver &gd, uint8_t* results, int dcs, int start_row, int rows, int start_col, int cols, int depth, int rcols,int rrows) {
    int free_buf=0;
    if(gd.lhs_start){
      free_buf = Check_For_Free_Buffer<int>(gd);
      Load_LHS_Data<int>(gd,free_buf,results,dcs,start_row,rows,start_col,cols,depth,rcols,rrows);
      Set_Results<int>(gd);
      Start_Transfer<int>(gd);
      gd.lhs_start=false;
    }else{
      bool gemm_done = Check_Done<int>(gd);
      free_buf = Check_For_Free_Buffer<int>(gd);
      if(free_buf!=-1){
        Load_LHS_Data<int>(gd,free_buf,results,dcs,start_row,rows,start_col,cols,depth,rcols,rrows);
        if(gemm_done){
          Store_Results<int>(gd);
          Set_Results<int>(gd);
          Start_Transfer<int>(gd);
        }
      }else{
        if(!gemm_done)Recieve_Results<int>(gd);
        Store_Results<int>(gd);
        if(gd.dID==gd.sID){
          free_buf = Check_For_Free_Buffer<int>(gd);
          Load_LHS_Data<int>(gd,free_buf,results,dcs,start_row,rows,start_col,cols,depth,rcols,rrows);
          Set_Results<int>(gd);
          Start_Transfer<int>(gd);
        }else{
          Set_Results<int>(gd);
          Start_Transfer<int>(gd);
          free_buf = Check_For_Free_Buffer<int>(gd);
          Load_LHS_Data<int>(gd,free_buf,results,dcs,start_row,rows,start_col,cols,depth,rcols,rrows);
        }
      }
    }
}

  //SECDA Added
template <typename Integer>
void DataHandleCompute(gemm_driver &gd, int dcs,int depth, int t_depth,int bpl2r, int bpl2c, int cols, uint8_params dst_params) {
  gd.t.layer_weight_tile=0;
  gd.t.layer_input_tile=0;
  uint8_t* results = dst_params.data;
#ifdef VM_ACC
  int acc_imax = 2048 * 16;
  int acc_wmax = 8192 * 16;
#else
  int acc_imax = 4096 * 16;
  int acc_wmax = 8192 * 16;
#endif

  int max_rows = acc_imax/t_depth;
  max_rows = max_rows - (max_rows % 4);
  int row_inc = std::min(std::min(bpl2r, max_rows), 4096);
  int max_cols = acc_wmax/t_depth;
  max_cols = max_cols - (max_cols % 4);
  int col_inc = std::min(std::min(bpl2c, max_cols), 4096);
  
  for (int o = 0; o < bpl2c;o +=col_inc){
    int os = std::min(col_inc, bpl2c - o);
    int rcols = std::min(col_inc, cols - o);
    gd.w_c = gd.wb_dex[gd.t.layer];
    
    Load_RHS_Data<int>(gd,o,os,depth,t_depth);
    for (int d = 0; d < t_depth; d +=t_depth) {
      int ds = std::min( t_depth, t_depth - d);
      for (int r = 0; r < bpl2r; r += row_inc) {
        int rs = std::min(row_inc, bpl2r - r);
        int rrows = std::min(row_inc, dst_params.rows - r);
        DataHandleComputeL1<int>(gd, results, dcs, r, rs, o, os, ds,rcols,rrows);
        gd.t.layer_input_tile++;
      }
    }

    while(gd.dID!=gd.rID){
      Recieve_Results<int>(gd);
      if(gd.dID!=gd.sID){Set_Results<int>(gd);Start_Transfer<int>(gd);}
      Store_Results<int>(gd);
    }
    dma_change_start<int>(gd.dma0,gd.dma1,gd.dma2,gd.dma3,0);
    gd.t.layer_weight_tile++;
  }
}


template <typename Integer>
void Entry(gemm_driver &gd, uint8_params lhs_params, uint8_params rhs_params, uint8_params dst_params){
  int depth = lhs_params.depth;
  int rows = dst_params.rows;
  int cols = dst_params.cols;
  int dcs = dst_params.rows;

  int temp_depth = tflite::RoundUp<16>(depth);
  int temp_cols =  tflite::RoundUp<2>(cols);
  int temp_rows =  tflite::RoundUp<4>(rows);

  // cout << "===========================" << endl;
  // cout << "Pre-ACC Info" << endl;
  // cout << "temp_depth: " << temp_depth << " depth: " << depth << endl;
  // cout << "temp_cols: " << temp_cols << " cols: " << dst_params.cols << endl;
  // cout << "temp_rows: " << temp_rows << " rows: " << dst_params.rows << endl;
  // cout << "old_dcs: " << temp_rows << " dcs: " << dcs << endl;
  // cout << "===========================" << endl;

  gd.dID=0;
  gd.sID=0;
  gd.rID=0;
  DataHandleCompute<int>(gd, dcs,depth,temp_depth,temp_rows,temp_cols,cols,dst_params);
  
  // if(true){
  //   uint8_t* res_pointer = dst_params.data;
  //   uint8_t res_pointerT[cols*rows];
  //   for (int i = 0;i<cols; ++i)for(int j=0; j<rows;++j)res_pointerT[i*rows+j]=res_pointer[j*cols+i];
  //   ofstream myfile;myfile.open ("uint_outs/"+std::to_string(gd.t.layer)+"_acc_res.csv");int index=0;
  //   for (int c = 0; c < rows;c++) {myfile << endl;for (int r = 0; r < cols;r++) {myfile << (int) res_pointer[index] << ","; index++; }}myfile.close();
  //   // ofstream myfile;myfile.open ("uint_outs/"+std::to_string(gd.t.layer)+"_acc_res.csv");uint8_t* res_pointer = dst_params.data;int index=0;
  //   // for (int c = 0; c < cols;c++) {myfile << endl;for (int r = 0; r < rows;r++) {myfile << (int) res_pointer[index] << ","; index++; }}myfile.close();
  // }
}
}
#endif
