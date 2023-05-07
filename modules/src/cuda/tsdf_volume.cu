#include "tsdf_volume.cuh"
#include "CudaUtils/cudaUtility.cuh"
#include "CudaUtils/cudaMath.cuh"
#include "camera_util.cuh"
#include "se3.cuh"

#include <limits>
#include <cstdlib>


namespace cuphoto {

constexpr i32 VOXEL_NEIGHBOURS = 8;
constexpr r32 BOUNDARY_EPS = 0.0001f;

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

__device__ int3 point_to_voxel(const float3 pt, const float3 voxel_size) {
    int3 voxel = make_int3(
        static_cast<i32>(floor(pt.x / voxel_size.x)),
        static_cast<i32>(floor(pt.y / voxel_size.y)),
        static_cast<i32>(floor(pt.z / voxel_size.z))
    );
    return voxel;
}

__device__ float3 voxel_center_in_grid(int3 voxel, const float3 voxel_size, const float3 offset = make_float3(0.f, 0.f, 0.f)) {
    float3 voxel_ct = make_float3(
        (static_cast<r32>(voxel.x) + 0.5f) * voxel_size.x + offset.x,
        (static_cast<r32>(voxel.y) + 0.5f) * voxel_size.y + offset.y,
        (static_cast<r32>(voxel.z) + 0.5f) * voxel_size.z + offset.z
    );
    return voxel_ct;
}

__device__ bool interpolate_trilinearly(const float3 vertex, const dim3 grid_size, const float3 voxel_size,
                                        i32 neighbor_ids[8], r32 coeffs[8]) {
    float3 boundary = make_float3(grid_size.x * voxel_size.x, 
                                  grid_size.y * voxel_size.y, 
                                  grid_size.z * voxel_size.z);

    float3 neighbor_pt = vertex;
    if (neighbor_pt.x < -BOUNDARY_EPS)
        neighbor_pt.x = 0.f;
    if (neighbor_pt.y < -BOUNDARY_EPS)
        neighbor_pt.y = 0.f;
    if (neighbor_pt.z < -BOUNDARY_EPS)
        neighbor_pt.z = 0.f;
    if ((neighbor_pt.x > boundary.x) && (neighbor_pt.x - boundary.x < BOUNDARY_EPS))
        neighbor_pt.x = boundary.x - BOUNDARY_EPS;
    if ((neighbor_pt.y > boundary.y) && (neighbor_pt.y - boundary.y < BOUNDARY_EPS))
        neighbor_pt.y = boundary.y - BOUNDARY_EPS;
    if ((neighbor_pt.z > boundary.z) && (neighbor_pt.z - boundary.z < BOUNDARY_EPS))
        neighbor_pt.z = boundary.z - BOUNDARY_EPS;
    
    int3 voxel = point_to_voxel(neighbor_pt, voxel_size);
    if (voxel.x < 0 && voxel.y < 0 && voxel.z < 0 &&
        voxel.x >= grid_size.x && voxel.y >= grid_size.y && voxel.z >= grid_size.z)
        return false;
    
    float3 voxel_ct = voxel_center_in_grid(voxel, voxel_size);

    int3 lower_bound = make_int3((neighbor_pt.x < voxel_ct.x) ? voxel.x - 1 : voxel.x, 
                                 (neighbor_pt.y < voxel_ct.y) ? voxel.y - 1 : voxel.y, 
                                 (neighbor_pt.z < voxel_ct.z) ? voxel.z - 1 : voxel.z);
    lower_bound.x = max(lower_bound.x, 0);
    lower_bound.y = max(lower_bound.y, 0);
    lower_bound.z = max(lower_bound.z, 0);
    float3 lower_ct = voxel_center_in_grid(lower_bound, voxel_size);
    float3 cubic_coord = neighbor_pt - lower_ct;
    cubic_coord = cubic_coord / voxel_size;
    i32 dx = 1;
    i32 dy = grid_size.x;
    i32 dz = grid_size.x * grid_size.y;
    neighbor_ids[0] = lower_bound.x + (lower_bound.y * grid_size.x) + (lower_bound.z * grid_size.x * grid_size.y);
    neighbor_ids[1] = neighbor_ids[0] + dx;
    neighbor_ids[2] = neighbor_ids[1] + dz;
    neighbor_ids[3] = neighbor_ids[0] + dz;
    neighbor_ids[4] = neighbor_ids[0] + dy;
    neighbor_ids[5] = neighbor_ids[1] + dy;
    neighbor_ids[6] = neighbor_ids[2] + dy;
    neighbor_ids[7] = neighbor_ids[3] + dy;

    coeffs[0] = (1 - cubic_coord.x) * (1 - cubic_coord.y) * (1 - cubic_coord.z);
    coeffs[1] = cubic_coord.x  * (1 - cubic_coord.y) * (1 - cubic_coord.z);
    coeffs[2] = cubic_coord.x * (1 - cubic_coord.y) * cubic_coord.z;
    coeffs[3] = (1 - cubic_coord.x) * (1 - cubic_coord.y) * cubic_coord.z;
    coeffs[4] = (1 - cubic_coord.x) * cubic_coord.y  * (1 - cubic_coord.z);
    coeffs[5] = cubic_coord.x * cubic_coord.y  * (1 - cubic_coord.z);
    coeffs[6] = (1 - cubic_coord.x) * cubic_coord.y  * cubic_coord.z;
    coeffs[7] = cubic_coord.x * cubic_coord.y  * cubic_coord.z;

    return true;
}

__global__ void init_voxel_transofrmation(TSDFVolume::TransformationVoxel::Ptr t_voxels, 
                                          const dim3 grid_size, const float3 voxel_size, 
                                          const float3 global_offset) {

    i32 vy = threadIdx.y + blockIdx.y * blockDim.y;
    i32 vz = threadIdx.z + blockIdx.z * blockDim.z;

    if (vy < grid_size.y && vz < grid_size.z) {

        i32 voxel_idx =  ((grid_size.x * grid_size.y) * vz) + (grid_size.x * vy);
        for (i32 vx = 0; vx < grid_size.x; vx++, voxel_idx++) {
            t_voxels[voxel_idx].translation.x = ((vx + 0.5f) * voxel_size.x) + global_offset.x;
            t_voxels[voxel_idx].translation.y = ((vy + 0.5f) * voxel_size.y) + global_offset.y;
            t_voxels[voxel_idx].translation.z = ((vz + 0.5f) * voxel_size.z) + global_offset.z;
            
            t_voxels[voxel_idx].quat_rotation.w = 1.0f;
            t_voxels[voxel_idx].quat_rotation.x = 0.0f;
            t_voxels[voxel_idx].quat_rotation.y = 0.0f;
            t_voxels[voxel_idx].quat_rotation.z = 0.0f;
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


__global__ void cu_tsdf_integrate(r32* voxel_distances, r32* voxel_weights,
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
    i32 voxel_idx = ((grid_size.x * grid_size.y) * vz ) + (grid_size.x * vy);
    for (i32 vx = 0; vx < grid_size.x; vx++, voxel_idx++) {
        float3 voxel_ct = t_voxels[voxel_idx].translation;
        int3 voxel_ct_px = world_to_pixel(voxel_ct, camera_pose, camera_k);
        if (voxel_ct_px.x < 0 || voxel_ct_px.x >= width || voxel_ct_px.y < 0 || voxel_ct_px.y >= height)
            continue;            
        const r32 depth_value = depth.ptr(voxel_ct_px.y)[voxel_ct_px.x];
        if (depth_value <= 0.0f || depth_value > 24.0)
            continue;
        float3 surface_pt = pixel_to_camera(voxel_ct_px, camera_k, depth_value);
        float3 voxel_cam = world_to_camera(voxel_ct, camera_pose);
        const r32 sdf = surface_pt.z - voxel_ct.z;
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
        }
        
    }
}


__global__ void cu_isosurface_transformation(const TSDFVolume::TransformationVoxel::Ptr t_voxels,
                                             const dim3 grid_size, const float3 voxel_size, 
                                             const float3 offset, const i32 num_pts,
                                             const float4 cam_rot_quat, const float3 cam_trans, 
                                             cudaPointCloud::Vertex::Ptr surface_points) {
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pts)
        return;
    float3 vertex = surface_points[idx].pos;
    
    vertex = vertex - offset;

    i32 neighbours[VOXEL_NEIGHBOURS];
    r32 coeffs[VOXEL_NEIGHBOURS];

    interpolate_trilinearly(vertex, grid_size, voxel_size, neighbours, coeffs);
    float3 t_iso_point = make_float3(0.f, 0.f, 0.f);
    t_iso_point = t_iso_point + (coeffs[0] * t_voxels[neighbours[0]].translation);
    t_iso_point = t_iso_point + (coeffs[1] * t_voxels[neighbours[1]].translation);
    t_iso_point = t_iso_point + (coeffs[2] * t_voxels[neighbours[2]].translation);
    t_iso_point = t_iso_point + (coeffs[3] * t_voxels[neighbours[3]].translation);
    t_iso_point = t_iso_point + (coeffs[4] * t_voxels[neighbours[4]].translation);
    t_iso_point = t_iso_point + (coeffs[5] * t_voxels[neighbours[5]].translation);
    t_iso_point = t_iso_point + (coeffs[6] * t_voxels[neighbours[6]].translation);
    t_iso_point = t_iso_point + (coeffs[7] * t_voxels[neighbours[7]].translation);

    SE3<r32> cam_pose(cam_rot_quat, cam_trans);

    t_iso_point = cam_pose * t_iso_point;

    surface_points[idx].pos = t_iso_point;
}

// =============================================================================

TSDFVolume::TSDFVolume(const TSDFVolumeConfig& tsdf_cfg) {
    cfg = tsdf_cfg;

    voxel_size = cfg.physical_size / cfg.voxel_grid_size;

    max_truncated_dist = 1.1f * length(voxel_size);

    const ui32 voxel_data_size = cfg.voxel_grid_size.x * cfg.voxel_grid_size.y * cfg.voxel_grid_size.z;

    CUDA(cudaMallocManaged(&voxel_distances, voxel_data_size * sizeof(r32), cudaMemAttachGlobal));

    CUDA(cudaMallocManaged(&voxel_weights, voxel_data_size * sizeof(r32), cudaMemAttachGlobal));

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

    dim3 blockDim2(1, 8, 8);
    dim3 gridDim2(1, iDivUp(cfg.voxel_grid_size.y, blockDim2.y), iDivUp(cfg.voxel_grid_size.z, blockDim2.z));
    init_voxel_transofrmation<<<gridDim2, blockDim2>>>(t_voxels, cfg.voxel_grid_size, voxel_size, cfg.global_offset);

    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::setup_data() -- failed setup transformation voxels data\n");
        return;
    }

    CUDA(cudaDeviceSynchronize());
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

    cu_tsdf_integrate<<<gridDim, blockDim>>>(voxel_distances, voxel_weights,
                                             cfg.voxel_grid_size, voxel_size, t_voxels,
                                             cfg.global_offset, max_truncated_dist, cfg.max_weight,
                                             camera_k, quat, trans, depth, color);
    
    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::integrate() -- failed integrate signed distance function on voxels data\n");
        return false;
    }
    
    CUDA(cudaDeviceSynchronize());

    return true;
}




bool TSDFVolume::integrate(const cudaPointCloud::Ptr cuda_pc, cv::cuda::PtrStepSzf depth,
                           const std::array<r64, 7>& camera_pose) {

    // TODO

    CUDA(cudaDeviceSynchronize());

    return true;
}




bool TSDFVolume::apply_isosurface_transformation(cudaPointCloud::Vertex::Ptr vertices,
                                                 const ui32 num_pts, 
                                                 const std::array<r64, 7>& camera_pose) {
    if (!vertices || num_pts <= 0) {
        LogError(LOG_CUDA "TSDFVolume::apply_isosurface_transformation() -- the given vertices is nullptr or num points is less than 0\n");
        return false;
    }
    float4 quat = make_float4(camera_pose[1], camera_pose[2], camera_pose[3], camera_pose[0]);
    float3 trans = make_float3(camera_pose[4], camera_pose[5], camera_pose[6]);
    dim3 blockDim(64, 1);
    dim3 gridDim(iDivUp(num_pts, blockDim.x), 1);

    cu_isosurface_transformation<<<gridDim, blockDim>>>(t_voxels, cfg.voxel_grid_size, voxel_size,
                                                        cfg.global_offset, num_pts, quat, trans, vertices);

    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "TSDFVolume::apply_isosurface_transformation() -- failed transform points of the surface\n");
        return false;
    }
    
    CUDA(cudaDeviceSynchronize());

    return true;
}



}