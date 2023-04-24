#include "tsdf_volume.cuh"
#include "frustrum.cuh"
#include "CudaUtils/cudaUtility.cuh"



namespace cuphoto {

__global__ void log_normal(const r32 sample, const r32 mean, const r32 var, r32& result)
{
    result = (-std::pow (sample - mean, 2) / (2 * var));
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

__global__ void test_kernel(Octree::Ptr octree, ui32* count_leaves, ui32* count_voxels) {
    const i32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 y = blockIdx.y * blockDim.y + threadIdx.y;
    *count_leaves = octree->count_leaves();
    *count_voxels = octree->count_voxels();
}


TSDFVolume::TSDFVolume(const TSDFVolumeConfig& tsdf_cfg, const i32 reserve_pool_size) {
    cfg = tsdf_cfg;
    octree = Octree::create_octree(reserve_pool_size, cfg.voxel_size);
}

TSDFVolume::~TSDFVolume() {
    Octree::free_octree(octree);
}

bool TSDFVolume::integrate(const cudaPointCloud::Ptr cuda_pc,
                           const std::array<r64, 7>& global_pose) {

    // TODO: add cuda point cloud transform with global pose
    const ui32 pool_size = cuda_pc->get_num_points() + 1;
    const float3 resolution_size = make_float3(cfg.resolution_size, cfg.resolution_size, cfg.resolution_size);
    octree->build_octree(pool_size, resolution_size);
   

    // TODO: add frustrum culling
    // float4 quat = make_float4(global_pose[1], global_pose[2], global_pose[3], global_pose[0]);
    // float3 trans = make_float3(global_pose[4], global_pose[5], global_pose[6]);
    // SE3<r32> se3_global_pose(quat, trans);





    CUDA(cudaDeviceSynchronize());

    return true;
}



void TSDFVolume::update_voxel() {
    // TODO
}


std::vector<int3> TSDFVolume::get_occupied_voxel_indices() const {
    // TODO
    return std::vector<int3>();
}

bool TSDFVolume::interpolate_trilinearly(const float3& pt, r32& dist) const {
    // TODO
    return true;
}

bool TSDFVolume::get_SDF_value(const float3& pt, r32& dist) const {
    // TODO
    return true;
}

bool TSDFVolume::fxn(const float3& pt, r32& value) const {
    // TODO
    return true;
}

bool TSDFVolume::gradient(const float3& pt, float3& grad) const {
    // TODO
    return true;
}


bool TSDFVolume::get_neighbors(const float3& query_pt, std::vector<OctreeNode::Ptr>& nodes, std::vector<float3>& centers) {
    int3 voxel_idx = make_int3(0, 0, 0);
    if (!get_voxel_index(query_pt, voxel_idx))
        return false;
    
    const float3 v_center = get_voxel_center(voxel_idx);    
    if (query_pt.x < v_center.x) 
        voxel_idx.x -= 1;
    if (query_pt.y < v_center.y)
        voxel_idx.y -= 1;
    if (query_pt.z < v_center.z) 
        voxel_idx.z -= 1;

    const i32 max_resol = cfg.resolution_size - 1;
    if (voxel_idx.x < 0 || voxel_idx.x >= max_resol || voxel_idx.y < 0 || voxel_idx.y >= max_resol || voxel_idx.z < 0 || voxel_idx.z >= max_resol)
        return false;

    for (i32 dx = 0; dx <= 1; dx++)
    {
        for (i32 dy = 0; dy <= 1; dy++)
        {
            for (i32 dz = 0; dz <= 1; dz++)
            {
                int3 dquery_voxel_ct = make_int3(voxel_idx.x + dx, voxel_idx.y + dy, voxel_idx.z + dz);
                float3 dvox_center = get_voxel_center(dquery_voxel_ct);
                const auto voxel_node = octree->query_voxel(dvox_center);
                if (voxel_node == nullptr)
                    return false;
                nodes.emplace_back(voxel_node);
                centers.emplace_back(dvox_center);
            }
        }
    }

    return true;
}

float3 TSDFVolume::get_voxel_center(const int3& pt_idx) {
    const r32 x_shift = cfg.voxel_size * 0.5;
    const r32 y_shift = cfg.voxel_size * 0.5;
    const r32 z_shift = cfg.voxel_size * 0.5;
    const r32 x_coord = pt_idx.x * cfg.voxel_size / cfg.resolution_size - x_shift;
    const r32 y_coord = pt_idx.y * cfg.voxel_size / cfg.resolution_size - y_shift;
    const r32 z_coord = pt_idx.z * cfg.voxel_size / cfg.resolution_size - z_shift;

    return make_float3(x_coord, y_coord, z_coord);
}   


bool TSDFVolume::get_voxel_index(const float3 pt, int3& ids) {
    const r32 x_shift = cfg.voxel_size * 0.5;
    const r32 y_shift = cfg.voxel_size * 0.5;
    const r32 z_shift = cfg.voxel_size * 0.5;

    const r32 res_size = static_cast<r32>(cfg.resolution_size);
    const i32 x_idx = (pt.x + x_shift) / cfg.voxel_size * res_size;
    const i32 y_idx = (pt.y + y_shift) / cfg.voxel_size * res_size;
    const i32 z_idx = (pt.z + z_shift) / cfg.voxel_size * res_size;

    const bool valid_voxel = (x_idx >= 0) && (y_idx >= 0) && (y_idx >= 0) &&
                             (x_idx < cfg.resolution_size) && (y_idx < cfg.resolution_size) && (z_idx < cfg.resolution_size);
    
    if (valid_voxel)
        ids = make_int3(x_idx, y_idx, z_idx);

    return valid_voxel;
}

}