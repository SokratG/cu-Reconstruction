#ifndef CUPHOTO_LIB_TENSORRT_UTILS_CUH
#define CUPHOTO_LIB_TENSORRT_UTILS_CUH

#include "types.cuh"

#include <opencv2/core/cuda/common.hpp>


namespace cuphoto {

static constexpr r32 MAX_INTENSITY = 255.f;


cudaError_t copy_tensor_normalization_RGB(const cv::cuda::PtrStepSzf input, r32* output, const ui32 output_width, 
                                          const ui32 output_height, const float2& norm_range, cudaStream_t stream,
                                          const r32 max_value = MAX_INTENSITY, size_t channel_stride=0, const bool chw = false);


cudaError_t copy_tensor_standardization_RGB(const cv::cuda::PtrStepSzf input, r32* output, const ui32 output_width, 
                                            const ui32 output_height, const float3& mean, const float3& std_dev, 
                                            cudaStream_t stream, size_t channel_stride=0, const bool chw = false);

  
};


#endif