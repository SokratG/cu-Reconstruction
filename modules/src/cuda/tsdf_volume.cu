#include "tsdf_volume.cuh"
#include "CudaUtils/cudaUtility.cuh"
#include "CudaUtils/cudaMath.cuh"
#include "camera_util.cuh"
#include "se3.cuh"

#include <limits>
#include <cstdlib>


namespace cuphoto {

__global__ void log_normal(const r32 sample, const r32 mean, const r32 var, r32& result)
{
    result = (-std::pow (sample - mean, 2) / (2 * var));
}

__global__ void set_array_to_value(r32* arr, const i32 size, const r32 value) {
    i32 idx = threadIdx.x + (blockIdx.x * blockDim.x );
    if (idx < size) {
        arr[idx] = value;
    }
}


__global__ void init_voxel_transofrmation(TSDFVolume::TransformationVoxel::Ptr t_voxels, 
                                          const dim3 grid_size, const float3 voxel_size, 
                                          const float3 global_offset) {

    i32 vy = threadIdx.y + blockIdx.y * blockDim.y;
    i32 vz = threadIdx.z + blockIdx.z * blockDim.z;

    if (vy < grid_size.y && vz < grid_size.z) {

        ui32 voxel_idx =  ((grid_size.x * grid_size.y) * vz ) + (grid_size.x * vy);
        for (i32 vx = 0; vx < grid_size.x; vx++ ) {
            t_voxels[voxel_idx].translation.x = ((vx + 0.5f ) * voxel_size.x) + global_offset.x;
            t_voxels[voxel_idx].translation.y = ((vy + 0.5f ) * voxel_size.y) + global_offset.y;
            t_voxels[voxel_idx].translation.z = ((vz + 0.5f ) * voxel_size.z) + global_offset.z;

            t_voxels[voxel_idx].rotation.x = 0.0f;
            t_voxels[voxel_idx].rotation.y = 0.0f;
            t_voxels[voxel_idx].rotation.z = 0.0f;

            voxel_idx++;
        }
    }
}

__host__ __device__ int2 reproject_point(const float3& pt, const r32 focal_x, const r32 focal_y,
                                         const r32 cx, const r32 cy, const i32 width, const i32 height)
{
    i32 u = (pt.x * focal_x / pt.z) + cx;
    i32 v = (pt.y * focal_y / pt.z) + cy;
    const bool valid = pt.z > 0 && u >= 0 && u < width && v >= 0 && v < height;
    if (!valid) {
        u = - 1;
        v = - 1; 
    }
    int2 px_coord = make_int2(u, v);
    return px_coord;
}


__global__ void cu_tsdf_integrate(r32* voxel_distances, r32* voxel_weights, uchar3* colors,
                                  dim3 grid_size, float3 voxel_size,
                                  TSDFVolume::TransformationVoxel::Ptr t_voxels,
                                  float3 offset, r32 max_truncated_dist, r32 max_weight,
                                  float4 camera_k, float4 quat, float3 trans,
                                  const cv::cuda::PtrStepSzf depth, const cv::cuda::PtrStepSzb rgb) {

    const i32 vy = threadIdx.y + blockIdx.y * blockDim.y;
    const i32 vz = threadIdx.z + blockIdx.z * blockDim.z;

    if (vy >= grid_size.y && vz >= grid_size.z)
        return;
    
    const ui32 width = depth.cols;
    const ui32 height = depth.rows;

    SE3<r32> camera_pose(quat, trans);
    SE3<r32> camera_pose_inv = camera_pose.inv();
    ui32 voxel_idx =  ((grid_size.x * grid_size.y) * vz ) + (grid_size.x * vy);
    for (i32 vx = 0; vx < grid_size.x; vx++, voxel_idx += 1) {
        float3 voxel_ct = offset + t_voxels[voxel_idx].translation;
        int3 voxel_ct_px = camera_to_pixel(voxel_ct, camera_k); // TODO: make world to pixel!
        if (voxel_ct_px.x < 0 || voxel_ct_px.x >= width || voxel_ct_px.y < 0 || voxel_ct_px.y >= height)
            continue;
        const r32 depth_value = depth.ptr(voxel_ct_px.y)[voxel_ct_px.x];
        if (depth_value <= 0.0)
            continue;
        float3 surface_pt = pixel_to_camera(voxel_ct_px, camera_k, depth_value);
        float3 voxel_cam = voxel_ct; // world_to_camera(voxel_ct, camera_pose_inv); // TODO: make world to camera!
        const r32 sdf = surface_pt.z - voxel_cam.z;
        if (sdf >= -max_truncated_dist) {
            r32 tsdf = sdf;
            if (sdf > 0)
                tsdf = fminf(sdf, max_truncated_dist);
            const r32 w = voxel_weights[voxel_idx];
            const r32 update_weight = 1.0f; // tune this param?
            const r32 new_w = w + update_weight;
            
            const r32 d = voxel_distances[voxel_idx];
            const r32 new_d = ((d * w) + (tsdf * update_weight)) / new_w;

            voxel_weights[voxel_idx] = new_w;
            voxel_distances[voxel_idx] = new_d;
            colors[voxel_idx] = ((uchar3*)rgb.ptr(voxel_ct_px.y))[voxel_ct_px.x];
        }
        
    }
}


__global__ void cu_isosurface_transformation(const TSDFVolume::TransformationVoxel::Ptr t_voxels,
                                            const dim3 grid_size, const float3 voxel_size, 
                                            const float3 physical_size,
                                            const float3 voxel_space_size, const float3 offset,
                                            const i32 num_pts, float3* surface_points) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pts)
        return;
    // TODO
}

// =============================================================================

TSDFVolume::TSDFVolume(const TSDFVolumeConfig& tsdf_cfg) {
    cfg = tsdf_cfg;

    voxel_size = cfg.physical_size / cfg.voxel_grid_size;

    max_truncated_dist = 1.1f * length(voxel_size);

    const ui32 voxel_data_size = cfg.voxel_grid_size.x * cfg.voxel_grid_size.y * cfg.voxel_grid_size.z;

    CUDA(cudaMallocManaged(&voxel_distances, voxel_data_size * sizeof(r32), cudaMemAttachGlobal));

    CUDA(cudaMallocManaged(&voxel_weights, voxel_data_size * sizeof(r32), cudaMemAttachGlobal));
    
    CUDA(cudaMallocManaged(&colors, voxel_data_size * sizeof(uchar3), cudaMemAttachGlobal));

    CUDA(cudaMallocManaged(&t_voxels, voxel_data_size * sizeof(TransformationVoxel), cudaMemAttachGlobal));

    CUDA(cudaMalloc(&custate, sizeof(curandState)));

    setup_data();
}

TSDFVolume::~TSDFVolume() {
    CUDA(cudaFree(custate));
    free_data();
}


void TSDFVolume::free_data() {
    CUDA(cudaFree(voxel_distances));
    CUDA(cudaFree(voxel_weights));
    CUDA(cudaFree(colors));
    CUDA(cudaFree(t_voxels));
}


void TSDFVolume::setup_data() {
    const ui32 voxel_data_size = cfg.voxel_grid_size.x * cfg.voxel_grid_size.y * cfg.voxel_grid_size.z;

    dim3 blockDim(64, 1, 1);
    dim3 gridDim(iDivUp(voxel_data_size, blockDim.x), 1, 1);
    set_array_to_value<<<gridDim, blockDim>>>(voxel_weights, voxel_data_size, 0.0f);
    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::setup_data() -- failed setup weights data\n");
        return;
    }

    set_array_to_value<<<gridDim, blockDim>>>(voxel_distances, voxel_data_size, max_truncated_dist);
    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::setup_data() -- failed setup distance data\n");
        return;
    }

    CUDA(cudaMemset(colors, voxel_data_size * 3, 0));
    dim3 blockDim2(1, 16, 16);
    dim3 gridDim2(1, iDivUp(cfg.voxel_grid_size.y, blockDim2.y), iDivUp(cfg.voxel_grid_size.z, blockDim2.z));
    init_voxel_transofrmation<<<gridDim2, blockDim2>>>(t_voxels, cfg.voxel_grid_size, voxel_size, cfg.global_offset);

    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::setup_data() -- failed setup transformation voxels data\n");
        return;
    }

    CUDA(cudaDeviceSynchronize());
}


bool TSDFVolume::integrate(const cudaPointCloud::Ptr cuda_pc,
                           cv::cuda::PtrStepSzf depth,
                           const std::array<r64, 7>& camera_pose) {

    // TODO
    

    CUDA(cudaDeviceSynchronize());

    return true;
}



bool TSDFVolume::integrate(const cv::cuda::PtrStepSzb color,
                           const  cv::cuda::PtrStepSzf depth,
                           const std::array<r64, 7>& camera_pose) {
    if (!depth || !color) {
        LogError(LOG_CUDA "TSDFVolume::integrate() -- depth or color map is null pointer\n");
		return false;
    }

    if (depth.rows == 0 || depth.cols == 0 || color.rows == 0 || color.cols == 0) {
        LogError(LOG_CUDA "TSDFVolume::integrate() -- depth width/height parameters are zero\n");
        return false;
    }

    float4 camera_k = make_float4(cfg.focal_x, cfg.focal_y, cfg.cx, cfg.cy);
    float4 quat = make_float4(camera_pose[1], camera_pose[2], camera_pose[3], camera_pose[0]);
    float3 trans = make_float3(camera_pose[4], camera_pose[5], camera_pose[6]);
    
    const dim3 blockDim(1, 8, 8);
	const dim3 gridDim(1, iDivUp(cfg.voxel_grid_size.y, blockDim.y), iDivUp(cfg.voxel_grid_size.z, blockDim.z));


    cu_tsdf_integrate<<<gridDim, blockDim>>>(voxel_distances, voxel_weights, colors,
                                             cfg.voxel_grid_size, voxel_size, t_voxels,
                                             cfg.global_offset, max_truncated_dist, cfg.max_weight,
                                             camera_k, quat, trans, depth, color);
    
    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::integrate() -- failed integrate tsd function on voxels data\n");
        return false;
    }
    
    CUDA(cudaDeviceSynchronize());

    return true;
}

bool TSDFVolume::apply_isosurface_transformation(const ui32 num_pts, float3* points) {
    // TODO
    return true;
}



}