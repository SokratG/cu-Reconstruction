#include "tensorrt_utils.cuh"

#include "CudaUtils/cudaUtility.cuh"

namespace cuphoto {

__device__ int3 get_indices(const size_t step_elem, const i32 x, const i32 y, const ui32 elem_size,
                            const ui32 width, const ui32 height, const ui32 stride) {
    if (stride > 0) {
        const i32 idx  = y * width + x;
        return make_int3(stride * 0 + idx, stride * 1 + idx, stride * 2 + idx);
    } else {
        const size_t step = step_elem / elem_size;
        const i32 mem_tid_step = step / width;
        const i32 idx  = y * step + (mem_tid_step * x);
        return make_int3(idx, idx + 1, idx + 2);
    }
}

__global__ void gpu_tensor_norm(const cv::cuda::PtrStepSzf input, r32* output, ui32 width, ui32 height, ui32 stride, 
                                r32 multiplier, r32 min_value)
{
    const i32 x = (blockDim.x * blockIdx.x) + threadIdx.x; // blockIdx blockDim threadIdx
    const i32 y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if(x >= width || y >= height)
        return;

    const float3 px = ((float3*)input.ptr(y))[x];

    const float3 rgb = make_float3(px.x, px.y, px.z);
    
    const int3 tid = get_indices(input.step, x, y, input.elemSize(), width, height, stride);

    output[tid.x] = rgb.x * multiplier + min_value;
    output[tid.y] = rgb.y * multiplier + min_value;
    output[tid.z] = rgb.z * multiplier + min_value;
}

__global__ void gpu_tensor_mean_stddev(const cv::cuda::PtrStepSzf input, r32* output, ui32 width, ui32 height, ui32 stride, 
                                       const float3& mean, const float3& std_dev)
{
    const i32 x = (blockDim.x * blockIdx.x) + threadIdx.x; // blockIdx blockDim threadIdx
    const i32 y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if(x >= width || y >= height)
        return;

    const float3 px = ((float3*)input.ptr(y))[x];

    const float3 rgb = make_float3(px.x, px.y, px.z);

    const int3 tid = get_indices(input.step, x, y, input.elemSize(), width, height, stride);

    output[tid.x] = (rgb.x - mean.x) / std_dev.x;
    output[tid.y] = (rgb.y - mean.y) / std_dev.y;
    output[tid.z] = (rgb.z - mean.z) / std_dev.z;
}

cudaError_t copy_tensor_normalization_RGB(const cv::cuda::PtrStepSzf input, r32* output, const ui32 output_width, 
                                          const ui32 output_height, const float2& norm_range, cudaStream_t stream,
                                          const r32 max_value, size_t channel_stride, const bool chw) {
    if(!input || !output)
        return cudaErrorInvalidDevicePointer;
    
    
    if(input.cols == 0 || input.rows == 0 || output_width == 0 || output_height == 0)
        return cudaErrorInvalidValue;
    
    if (chw) {
        if (channel_stride == 0)
            channel_stride = output_width * output_height;
    }

    const r32 multiplier = (norm_range.y - norm_range.x) / max_value;

    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(output_width, blockDim.x), iDivUp(output_height, blockDim.y), 1);
    
    gpu_tensor_norm<<<gridDim, blockDim, 1, stream>>>(input, output, output_width, output_height, channel_stride, multiplier, norm_range.x);

    return CUDA(cudaGetLastError());
}

cudaError_t copy_tensor_standardization_RGB(const cv::cuda::PtrStepSzf input, r32* output, const ui32 output_width, 
                                            const ui32 output_height, const float3& mean, const float3& std_dev, 
                                            cudaStream_t stream, size_t channel_stride, const bool chw) {
    if(!input || !output)
        return cudaErrorInvalidDevicePointer;


    if(input.cols == 0 || input.rows == 0 || output_width == 0 || output_height == 0)
        return cudaErrorInvalidValue;

    if (chw) {
        if (channel_stride == 0)
            channel_stride = output_width * output_height;
    }

    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(output_width, blockDim.x), iDivUp(output_height, blockDim.y), 1);

    gpu_tensor_mean_stddev<<<gridDim, blockDim, 1, stream>>>(input, output, output_width, output_height, channel_stride, mean, std_dev);

    return CUDA(cudaGetLastError());

}

    
};