#ifndef CUPHOTO_LIB_UTIL_CUH
#define CUPHOTO_LIB_UTIL_CUH

#include "types.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <opencv2/core/cuda/common.hpp>


namespace cuphoto {

static bool cudaHandleError(cudaError_t err, const ch* file, i32 line)
{
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        return false;
    }
    return true;
}

void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<sh16> src,
                                   const i32 width, const i32 height);

cudaError_t disparity_to_depth(const cv::cuda::PtrStepSz<sh16> input_data, r32* output_data, 
                               const r32 focal, const r32 baseline, const r32 depth_scale = 1.0);


cudaError_t setup_cuda_rand_state(curandState* cu_rand_state);

cudaError_t generate_random_float3(curandState* cu_rand_state, const r32 scale, float3& rand_vec);

__device__ bool near_equal(const r32 a, const r32 b);

__device__ bool get_trilinear_elements(const float3 point, 
                                       const dim3 grid_size,
                                       const float3 voxel_size,
                                       i32* indices,
                                       r32* coefficients); 

};

#endif