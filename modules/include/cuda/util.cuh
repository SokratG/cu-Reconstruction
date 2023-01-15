#ifndef CUREC_LIB_UTIL_CUH
#define CUREC_LIB_UTIL_CUH

#include "types.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

#include <opencv2/core/cuda/common.hpp>

namespace curec {

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


};

#endif