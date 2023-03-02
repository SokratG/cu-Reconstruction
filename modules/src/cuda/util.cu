#include "util.cuh"
#include "CudaUtils/cudaUtility.cuh"


namespace cuphoto {

__host__ inline int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


__global__ void normalizeUsingWeightKernel32F(const cv::cuda::PtrStepf weight, 
                                              cv::cuda::PtrStep<sh16> src,
                                              const i32 width, const i32 height)
{
    i32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
    i32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height)
    {
        constexpr r32 WEIGHT_EPS = 1e-5f;
        const short3 v = ((short3*)src.ptr(y))[x];
        r32 w = weight.ptr(y)[x];
        ((short3*)src.ptr(y))[x] = make_short3(static_cast<sh16>(v.x / (w + WEIGHT_EPS)),
                                               static_cast<sh16>(v.y / (w + WEIGHT_EPS)),
                                               static_cast<sh16>(v.z / (w + WEIGHT_EPS)));
    }
}


__global__ void disparity_to_depth_kernel(const cv::cuda::PtrStepSz<sh16> input_data, r32* output_data,
                                          const ui32 width, const ui32 height,
                                          const r32 focal, const r32 baseline, const r32 depth_scale)
{
    i32 x = (blockIdx.x * blockDim.x) + threadIdx.x;
    i32 y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= width && y >= height)
        return;
    
    const sh16 v = ((sh16*)input_data.ptr(y))[x];

    if (v == 0)
        return;
    
    const i32 tid = y * width + x;

    const r32 disp_val = static_cast<r32>(v);
    output_data[tid] = ((focal * baseline) / disp_val) * depth_scale;
}


void normalizeUsingWeightMapGpu32F(const cv::cuda::PtrStepf weight, cv::cuda::PtrStep<sh16> src,
                                   const i32 width, const i32 height)
{
    dim3 blockDim(8, 8);
    dim3 gridDim(divUp(width, blockDim.x), divUp(height, blockDim.y));
    normalizeUsingWeightKernel32F<<<gridDim, blockDim>>> (weight, src, width, height);
}



cudaError_t disparity_to_depth(const cv::cuda::PtrStepSz<sh16> input_data, r32* output_data, 
                               const r32 focal, const r32 baseline, const r32 depth_scale) {
    if (!input_data || !output_data)
        return cudaErrorInvalidDevicePointer;


    if(input_data.cols == 0 || input_data.rows == 0)
        return cudaErrorInvalidValue;
    
    dim3 blockDim(8, 8, 1);
    dim3 gridDim(divUp(input_data.cols, blockDim.x), divUp(input_data.rows, blockDim.y), 1);
    
    disparity_to_depth_kernel<<<gridDim, blockDim, 1>>>(input_data, output_data, 
                                                        input_data.cols, input_data.rows, 
                                                        focal, baseline, depth_scale);
    
    return CUDA(cudaGetLastError());
}

};
