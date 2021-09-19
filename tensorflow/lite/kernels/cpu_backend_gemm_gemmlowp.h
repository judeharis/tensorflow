/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_GEMMLOWP_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_GEMMLOWP_H_

#ifndef TFLITE_WITH_RUY

#include <cstdint>
#include <type_traits>

#include "public/gemmlowp.h"
#include "tensorflow/lite/experimental/ruy/ruy.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"


#include "tensorflow/lite/examples/label_image_secda/gemm_driver.h"

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

template <typename DstScalar>
struct GemmlowpSaturatingCastStage {};

template <>
struct GemmlowpSaturatingCastStage<std::uint8_t> {
  using Type = gemmlowp::OutputStageSaturatingCastToUint8;
};

template <>
struct GemmlowpSaturatingCastStage<std::int8_t> {
  using Type = gemmlowp::OutputStageSaturatingCastToInt8;
};

template <>
struct GemmlowpSaturatingCastStage<std::int16_t> {
  using Type = gemmlowp::OutputStageSaturatingCastToInt16;
};

template <typename DstScalar>
struct GemmlowpBitDepthParams {};

template <>
struct GemmlowpBitDepthParams<std::uint8_t> {
  using Type = gemmlowp::L8R8WithLhsNonzeroBitDepthParams;
};

template <>
struct GemmlowpBitDepthParams<std::int8_t> {
  using Type = gemmlowp::SignedL8R8WithLhsNonzeroBitDepthParams;
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingGemmlowp {};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct GemmImplUsingGemmlowp<
    LhsScalar, RhsScalar, AccumScalar, DstScalar,
    QuantizationFlavor::kIntegerWithUniformMultiplier> {
  static_assert(std::is_same<LhsScalar, RhsScalar>::value, "");
  static_assert(std::is_same<AccumScalar, std::int32_t>::value, "");
  using SrcScalar = LhsScalar;

  static void Run(
      const MatrixParams<SrcScalar>& lhs_params, const SrcScalar* lhs_data,
      const MatrixParams<SrcScalar>& rhs_params, const SrcScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<std::int32_t, DstScalar,
                       QuantizationFlavor::kIntegerWithUniformMultiplier>&
          params,
      CpuBackendContext* context) {
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::RowMajor>
        gemmlowp_lhs(lhs_data, lhs_params.rows, lhs_params.cols);
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::ColMajor>
        gemmlowp_rhs(rhs_data, rhs_params.rows, rhs_params.cols);
    gemmlowp::MatrixMap<DstScalar, gemmlowp::MapOrder::ColMajor> gemmlowp_dst(
        dst_data, dst_params.rows, dst_params.cols);

    using ColVectorMap =
        gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>;
    ColVectorMap bias_vector(params.bias, lhs_params.rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponent scale_stage;
    scale_stage.result_offset_after_shift = dst_params.zero_point;
    scale_stage.result_fixedpoint_multiplier = params.multiplier_fixedpoint;
    scale_stage.result_exponent = params.multiplier_exponent;
    using SaturatingCastStageType =
        typename GemmlowpSaturatingCastStage<DstScalar>::Type;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = params.clamp_min;
    clamp_stage.max = params.clamp_max;
    SaturatingCastStageType saturating_cast_stage;
    auto output_pipeline = std::make_tuple(bias_addition_stage, scale_stage,
                                           clamp_stage, saturating_cast_stage);
    using BitDepthParams = typename GemmlowpBitDepthParams<SrcScalar>::Type;
    gemmlowp::GemmWithOutputPipeline<SrcScalar, DstScalar, BitDepthParams>(
        context->gemmlowp_context(), gemmlowp_lhs, gemmlowp_rhs, &gemmlowp_dst,
        -lhs_params.zero_point, -rhs_params.zero_point, output_pipeline);
  }

  // SECDA: Added
  static void Run2(gemm_driver &gd,
      const MatrixParams<SrcScalar>& lhs_params, const SrcScalar* lhs_data,
      const MatrixParams<SrcScalar>& rhs_params, const SrcScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<std::int32_t, DstScalar,QuantizationFlavor::kIntegerWithUniformMultiplier>&params,
      CpuBackendContext* context) {
          
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::RowMajor>
        gemmlowp_lhs(lhs_data, lhs_params.rows, lhs_params.cols);
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::ColMajor>
        gemmlowp_rhs(rhs_data, rhs_params.rows, rhs_params.cols);
    gemmlowp::MatrixMap<DstScalar, gemmlowp::MapOrder::ColMajor> gemmlowp_dst(
        dst_data, dst_params.rows, dst_params.cols);

    prf_start(1);
    int width = rhs_params.cols;
    int w = ((width + 3) - ((width + 3) % 4));
    int depth = lhs_params.cols;
    int d = ((depth + 15) - ((depth + 15) % 16));
    int d2 = depth*2;
    int d3 = depth*3;
    int d4 = depth*4;

    int s_need = w*d/4 +1;
    uint8_t inb0[s_need];
    uint8_t inb1[s_need];
    uint8_t inb2[s_need];
    uint8_t inb3[s_need];
    int i_c =0;
    int sums_curr=0;
    gd.rhs_offset = -rhs_params.zero_point;
    gd.lhs_offset = -lhs_params.zero_point;

    int in_sum1[w/4];
    int in_sum2[w/4];
    int in_sum3[w/4];
    int in_sum4[w/4];

    const uint8_t* rhs_d = reinterpret_cast<const uint8_t*> (rhs_data);
    int dm=0;
    for(int i=0; i<w/4;i++){
        int id = i*d4;
        int i0= id;
        int i1= id+ depth;
        int i2= id+ d2;
        int i3= id+ d3;
        int  ss0 =0;
        int  ss1 =0;
        int  ss2 =0;
        int  ss3 =0;

#ifdef ACC_NEON
        dm= d-16;
        uint8x16_t tmp0;
        uint8x16_t tmp1;
        uint8x16_t tmp2;
        uint8x16_t tmp3;
        uint32x4_t  tmp0_2;
        uint32x4_t  tmp1_2;
        uint32x4_t  tmp2_2;
        uint32x4_t  tmp3_2;
        uint32x2_t  tmp0_3;
        uint32x2_t  tmp1_3;
        uint32x2_t  tmp2_3;
        uint32x2_t  tmp3_3;
        uint32x2_t  tmp0_4 =vdup_n_u32(0);
        uint32x2_t  tmp1_4 =vdup_n_u32(0);
        uint32x2_t  tmp2_4 =vdup_n_u32(0);
        uint32x2_t  tmp3_4 =vdup_n_u32(0);

        for(int j=0; j<dm;j+=16){
            tmp0= vld1q_u8(rhs_d + i0 + j);
            tmp1= vld1q_u8(rhs_d + i1 + j);
            tmp2= vld1q_u8(rhs_d + i2 + j);
            tmp3= vld1q_u8(rhs_d + i3 + j);
            tmp0_2 = vpaddlq_u16(vpaddlq_u8(tmp0));
            tmp1_2 = vpaddlq_u16(vpaddlq_u8(tmp1));
            tmp2_2 = vpaddlq_u16(vpaddlq_u8(tmp2));
            tmp3_2 = vpaddlq_u16(vpaddlq_u8(tmp3));

            tmp0_3 = vadd_u32(vget_high_u32(tmp0_2),vget_low_u32(tmp0_2));
            tmp1_3 = vadd_u32(vget_high_u32(tmp1_2),vget_low_u32(tmp1_2));
            tmp2_3 = vadd_u32(vget_high_u32(tmp2_2),vget_low_u32(tmp2_2));
            tmp3_3 = vadd_u32(vget_high_u32(tmp3_2),vget_low_u32(tmp3_2));
            tmp0_4 = vadd_u32(tmp0_4,tmp0_3);
            tmp1_4 = vadd_u32(tmp1_4,tmp1_3);
            tmp2_4 = vadd_u32(tmp2_4,tmp2_3);
            tmp3_4 = vadd_u32(tmp3_4,tmp3_3);
            vst1q_u8(inb0+i_c, tmp0);
            vst1q_u8(inb1+i_c, tmp1);
            vst1q_u8(inb2+i_c, tmp2);
            vst1q_u8(inb3+i_c, tmp3);
            i_c+=16;
        }
        uint32_t tmp0_s[2];
        uint32_t tmp1_s[2];
        uint32_t tmp2_s[2];
        uint32_t tmp3_s[2];
        vst1_u32(tmp0_s,tmp0_4);
        vst1_u32(tmp1_s,tmp1_4);
        vst1_u32(tmp2_s,tmp2_4);
        vst1_u32(tmp3_s,tmp3_4);
        ss0 += tmp0_s[0]+tmp0_s[1];
        ss1 += tmp1_s[0]+tmp1_s[1];
        ss2 += tmp2_s[0]+tmp2_s[1];
        ss3 += tmp3_s[0]+tmp3_s[1];
#endif
        for(int j=dm; j<d;j++){
            if (j<depth){
                unsigned char w0 = rhs_data[i0+j];
                unsigned char w1 = rhs_data[i1+j];
                unsigned char w2 = rhs_data[i2+j];
                unsigned char w3 = rhs_data[i3+j];
                ss0+=w0;
                ss1+=w1;
                ss2+=w2;
                ss3+=w3;
                inb0[i_c]=w0;
                inb1[i_c]=w1;
                inb2[i_c]=w2;
                inb3[i_c++]=w3;
            }else{
                inb0[i_c]=0;
                inb1[i_c]=0;
                inb2[i_c]=0;
                inb3[i_c++]=0;
            }
        }
        in_sum1[sums_curr]=(ss0);
        in_sum2[sums_curr]=(ss1);
        in_sum3[sums_curr]=(ss2);
        in_sum4[sums_curr++]=(ss3);
    }
    gd.in_id=0;
    unsigned int* inb_0 = reinterpret_cast<unsigned int*> (inb0);
    unsigned int* inb_1 = reinterpret_cast<unsigned int*> (inb1);
    unsigned int* inb_2 = reinterpret_cast<unsigned int*> (inb2);
    unsigned int* inb_3 = reinterpret_cast<unsigned int*> (inb3);
    gd.inb_0=inb_0;
    gd.inb_1=inb_1;
    gd.inb_2=inb_2;
    gd.inb_3=inb_3;

    gd.in_sum_len=sums_curr;
    gd.in_sum1=in_sum1;
    gd.in_sum2=in_sum2;
    gd.in_sum3=in_sum3;
    gd.in_sum4=in_sum4;
    gd.bias = std::vector<int>(lhs_params.rows);
    for(int i=0;i<lhs_params.rows;i++)gd.bias[i] = params.bias[i];
    gd.ra = dst_params.zero_point;
    gd.rf = params.multiplier_fixedpoint;
    gd.re = params.multiplier_exponent;
    prf_end(1,gd.t.acctime);

    prf_start(2);
    // gemmlowp::GemmWithOutputPipeline2<SrcScalar, DstScalar>(gd,context->gemmlowp_context(), gemmlowp_lhs, gemmlowp_rhs,&gemmlowp_dst);
    gemmlowp::MultiThreadGemm2<SrcScalar, DstScalar>(gd,context->gemmlowp_context(), gemmlowp_lhs, gemmlowp_rhs,&gemmlowp_dst);
    prf_end(2,gd.t.acctime);
  }
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
struct GemmImplUsingGemmlowp<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                             QuantizationFlavor::kIntegerWithPerRowMultiplier> {
  static_assert(std::is_same<LhsScalar, RhsScalar>::value, "");
  static_assert(std::is_same<AccumScalar, std::int32_t>::value, "");
  using SrcScalar = LhsScalar;

  static void Run(
      const MatrixParams<SrcScalar>& lhs_params, const SrcScalar* lhs_data,
      const MatrixParams<SrcScalar>& rhs_params, const SrcScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<std::int32_t, DstScalar,
                       QuantizationFlavor::kIntegerWithPerRowMultiplier>&
          params,
      CpuBackendContext* context) {
    // gemmlowp support for this per-channel path is limited to NEON.
    // We fall back to ruy outside of NEON.
#ifdef GEMMLOWP_NEON
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::RowMajor>
        gemmlowp_lhs(lhs_data, lhs_params.rows, lhs_params.cols);
    gemmlowp::MatrixMap<const SrcScalar, gemmlowp::MapOrder::ColMajor>
        gemmlowp_rhs(rhs_data, rhs_params.rows, rhs_params.cols);
    gemmlowp::MatrixMap<DstScalar, gemmlowp::MapOrder::ColMajor> gemmlowp_dst(
        dst_data, dst_params.rows, dst_params.cols);

    using ColVectorMap =
        gemmlowp::VectorMap<const int32, gemmlowp::VectorShape::Col>;
    ColVectorMap bias_vector(params.bias, lhs_params.rows);
    gemmlowp::OutputStageBiasAddition<ColVectorMap> bias_addition_stage;
    bias_addition_stage.bias_vector = bias_vector;
    gemmlowp::OutputStageScaleInt32ByFixedPointAndExponentPC<
        gemmlowp::VectorShape::Col>
        scale_stage;
    scale_stage.result_offset_after_shift = dst_params.zero_point;
    scale_stage.result_fixedpoint_multiplier =
        ColVectorMap(params.multiplier_fixedpoint_perchannel, dst_params.rows);
    scale_stage.result_exponent =
        ColVectorMap(params.multiplier_exponent_perchannel, dst_params.rows);
    using SaturatingCastStageType =
        typename GemmlowpSaturatingCastStage<DstScalar>::Type;
    gemmlowp::OutputStageClamp clamp_stage;
    clamp_stage.min = params.clamp_min;
    clamp_stage.max = params.clamp_max;
    SaturatingCastStageType saturating_cast_stage;
    auto output_pipeline = std::make_tuple(bias_addition_stage, scale_stage,
                                           clamp_stage, saturating_cast_stage);
    using BitDepthParams = typename GemmlowpBitDepthParams<SrcScalar>::Type;
    gemmlowp::GemmWithOutputPipeline<SrcScalar, DstScalar, BitDepthParams>(
        context->gemmlowp_context(), gemmlowp_lhs, gemmlowp_rhs, &gemmlowp_dst,
        -lhs_params.zero_point, -rhs_params.zero_point, output_pipeline);
#else
    GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                     QuantizationFlavor::kIntegerWithPerRowMultiplier>::
        Run(lhs_params, lhs_data, rhs_params, rhs_data, dst_params, dst_data,
            params, context);
#endif
  }
};

}  // namespace detail
}  // namespace cpu_backend_gemm
}  // namespace tflite

#endif  // not TFLITE_WITH_RUY

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_GEMMLOWP_H_
