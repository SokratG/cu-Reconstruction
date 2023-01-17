#include "util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/types.hpp>


namespace curec {

__host__ inline int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__global__ void normalizeUsingWeightKernel32F(const cv::cuda::PtrStepf weight, 
                                              cv::cuda::PtrStep<sh16> src,
                                              const i32 width, const i32 height)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height)
    {
        constexpr float WEIGHT_EPS = 1e-5f;
        const short3 v = ((short3*)src.ptr(y))[x];
        float w = weight.ptr(y)[x];
        ((short3*)src.ptr(y))[x] = make_short3(static_cast<short>(v.x / (w + WEIGHT_EPS)),
                                               static_cast<short>(v.y / (w + WEIGHT_EPS)),
                                               static_cast<short>(v.z / (w + WEIGHT_EPS)));
    }
}


void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<sh16> src,
                                   const i32 width, const i32 height)
{
    dim3 threads(32, 32);
    dim3 grid(divUp(width, threads.x), divUp(height, threads.y));
    normalizeUsingWeightKernel32F<<<grid, threads>>> (weight, src, width, height);
}

};
