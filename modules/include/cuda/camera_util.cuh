#ifndef CUPHOTO_LIB_CAMERA_UTIL_CUH
#define CUPHOTO_LIB_CAMERA_UTIL_CUH

#include "types.cuh"
#include "se3.cuh"
#include <cuda_runtime.h>

namespace cuphoto {

__host__ __device__ int3 camera_to_pixel(const float3 cam_pt, const float4 camera_k);

__host__ __device__ float3 pixel_to_camera(const int3 px_pt, const float4 camera_k, const r32 depth = 1.0);

__host__ __device__ float3 world_to_camera(const float3 world_pt, const SE3<r32>& camera_pose);

__host__ __device__ float3 camera_to_world(const float3 cam_pt, const SE3<r32>& camera_pose_inv);

__host__ __device__ int3 world_to_pixel(const float3 world_pt, const SE3<r32>& camera_pose, const float4 camera_k);

__host__ __device__ float3 pixel_to_world(const int3 px_pt, const SE3<r32>& camera_pose_inv, const float4 camera_k, const r32 depth = 1.0);

}


#endif