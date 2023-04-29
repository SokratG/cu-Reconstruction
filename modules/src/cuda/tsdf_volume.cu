#include "tsdf_volume.cuh"
#include "frustrum.cuh"
#include "CudaUtils/cudaUtility.cuh"

#include <limits>
#include <cstdlib>


namespace cuphoto {

__global__ void log_normal(const r32 sample, const r32 mean, const r32 var, r32& result)
{
    result = (-std::pow (sample - mean, 2) / (2 * var));
}


float3 random_float3(const r32 scale) {
    const r32 x = (static_cast<r32>(std::rand()) / static_cast<r32>(RAND_MAX)) * scale;
    const r32 y = (static_cast<r32>(std::rand()) / static_cast<r32>(RAND_MAX)) * scale;
    const r32 z = (static_cast<r32>(std::rand()) / static_cast<r32>(RAND_MAX)) * scale;
    return make_float3(x, y, z);
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


TSDFVolume::TSDFVolume(const TSDFVolumeConfig& tsdf_cfg, const i32 reserve_pool_size) {
    cfg = tsdf_cfg;
    cudaMalloc(&custate, sizeof(curandState));
    octree = Octree::create_octree(reserve_pool_size, cfg.voxel_size);
}

TSDFVolume::~TSDFVolume() {
    cudaFree(custate);
    Octree::free_octree(octree);
}

bool TSDFVolume::integrate(const cudaPointCloud::Ptr cuda_pc,
                           const std::array<r64, 7>& global_pose) {

    // TODO: add cuda point cloud transform with global pose
    const float3 resolution_size = make_float3(cfg.resolution_size, cfg.resolution_size, cfg.resolution_size);
    octree->build_octree(resolution_size);
            
    const ui32 num_pts = cuda_pc->get_num_points();
    const r32 ratio_size = cfg.voxel_size / cfg.resolution_size;
    auto& cuda_pool_allocator = octree->allocator();
    for (ui32 idx = 0; idx < num_pts; ++idx) {
        cudaPointCloud::Vertex::Ptr v = cuda_pc->get_vertex(idx);
        for (i32 rsplit = 0; rsplit < cfg.num_rand_split; ++rsplit) {
            const r32 scale = static_cast<r32>(std::rand()) / static_cast<r32>(RAND_MAX) * cfg.voxel_size;
            float3 noise = random_float3(scale);
            float3 query_pt = v->pos + noise;
            OctreeNode::Ptr voxel = octree->query_voxel(query_pt);
            if (voxel != nullptr) {
                while (voxel->voxel_size() > ratio_size) {              
                    voxel->split(cuda_pool_allocator);
                    voxel = voxel->query_voxel(query_pt);
                }
            }
        }
    }

    // TODO: add frustrum culling
    // float4 quat = make_float4(global_pose[1], global_pose[2], global_pose[3], global_pose[0]);
    // float3 trans = make_float3(global_pose[4], global_pose[5], global_pose[6]);
    // SE3<r32> se3_global_pose(quat, trans);

    distance_to_voxel(cuda_pc);
    

    CUDA(cudaDeviceSynchronize());

    return true;
}



void TSDFVolume::distance_to_voxel(const cudaPointCloud::Ptr cuda_pc) {
    // TODO
    const ui32 leaves_num = octree->count_leaves();
    CudaObjectPoolAllocator<OctreeNode::OctreeListNode> oct_list_pa;
    oct_list_pa.reserve(leaves_num + 1);
    
    OctreeNode::OctreeListNode::Ptr oct_list_head = nullptr;
    octree->leaves(oct_list_head, oct_list_pa);
    OctreeNode::PPtr voxels = nullptr;
    CUDA(cudaMallocManaged(&voxels, leaves_num * sizeof(OctreeNode::Ptr), cudaMemAttachGlobal));
    for (i32 idx = 0; oct_list_head != nullptr; ++idx ) {
        voxels[idx] = oct_list_head->oct_node;
        oct_list_head = oct_list_head->next;
    }



    if (voxels != nullptr)
        CUDA(cudaFree(voxels));
}


std::vector<int3> TSDFVolume::get_occupied_voxel_indices() const {
    const i32 leaves_num = octree->count_leaves();
    CudaObjectPoolAllocator<OctreeNode::OctreeListNode> octree_lst_allocator;
    octree_lst_allocator.reserve(leaves_num);
    OctreeNode::OctreeListNode::Ptr list_leaves;
    const bool res = octree->leaves(list_leaves, octree_lst_allocator);

    if (!res)
        return std::vector<int3>();

    OctreeNode::OctreeListNode::Ptr current_voxel = list_leaves;
    std::vector<int3> indices;
    for (i32 lidx = 0; lidx < leaves_num && current_voxel->next != nullptr; ++lidx) {
        const r32 dist = current_voxel->oct_node->distance();
        const r32 weight = current_voxel->oct_node->weight();
        if (weight > 0 && std::abs(dist) < 1.0) {
            const float3 center = current_voxel->oct_node->center();
            int3 ids = make_int3(0, 0, 0);
            const bool res = get_voxel_index(center, ids);
            if (res)
                indices.emplace_back(ids);
        }
        current_voxel = current_voxel->next;
    }


    return indices;
}

bool TSDFVolume::interpolate_trilinearly(const float3& query_pt, r32& dist) const {
    
    int3 vidx = make_int3(0, 0, 0);
    bool res = get_voxel_index(query_pt, vidx);
    const i32 max_resol = cfg.resolution_size - 1;
    if (!res || vidx.x < 0 || vidx.x >= max_resol || vidx.y < 0 || vidx.y >= max_resol || vidx.z < 0 || vidx.z >= max_resol) {
        dist = std::numeric_limits<r32>::quiet_NaN();
        return false;
    }
    float3 vcenter = get_voxel_center(vidx);
    if (query_pt.x < vcenter.x) 
        vidx.x -= 1;
    if (query_pt.y < vcenter.y)
        vidx.y -= 1;
    if (query_pt.z < vcenter.z) 
        vidx.z -= 1;
    
    vcenter = get_voxel_center(vidx);
    float3 voxel_cx = get_voxel_center(make_int3(vidx.x + 1, vidx.y, vidx.z));
    float3 voxel_cy = get_voxel_center(make_int3(vidx.x, vidx.y + 1, vidx.z));
    float3 voxel_cz = get_voxel_center(make_int3(vidx.x, vidx.y, vidx.z + 1));
    float3 voxel_cxy = get_voxel_center(make_int3(vidx.x + 1, vidx.y + 1, vidx.z));
    float3 voxel_cxz = get_voxel_center(make_int3(vidx.x + 1, vidx.y, vidx.z + 1));
    float3 voxel_cyz = get_voxel_center(make_int3(vidx.x, vidx.y + 1, vidx.z + 1));
    float3 voxel_cxyz = get_voxel_center(make_int3(vidx.x + 1, vidx.y + 1, vidx.z + 1));

    const r32 ratio = cfg.voxel_size / cfg.resolution_size;
    const r32 a = (query_pt.x - vcenter.x) * ratio;
    const r32 b = (query_pt.y - vcenter.y) * ratio;
    const r32 c = (query_pt.z - vcenter.z) * ratio;

    const OctreeNode::Ptr oct_voxel = octree->query_voxel(vcenter);
    const OctreeNode::Ptr oct_voxel_x = octree->query_voxel(voxel_cx);
    const OctreeNode::Ptr oct_voxel_y = octree->query_voxel(voxel_cy);
    const OctreeNode::Ptr oct_voxel_z = octree->query_voxel(voxel_cz);
    const OctreeNode::Ptr oct_voxel_xy = octree->query_voxel(voxel_cxy);
    const OctreeNode::Ptr oct_voxel_xz = octree->query_voxel(voxel_cxz);
    const OctreeNode::Ptr oct_voxel_yz = octree->query_voxel(voxel_cyz);
    const OctreeNode::Ptr oct_voxel_xyz = octree->query_voxel(voxel_cxyz);

    dist = oct_voxel->distance() * (1 - a) * (1 - b) * (1 - c) +
           oct_voxel_x->distance() * a * (1 - b) * (1 - c) +
           oct_voxel_y->distance() * (1 - a) * b * (1 - c) +
           oct_voxel_z->distance() * (1 - a) * (1 - b) * c +
           oct_voxel_xy->distance() * a * b * (1 - c) +
           oct_voxel_xz->distance() * a * (1 - b) * c +
           oct_voxel_yz->distance() * (1 - a) * b * c +
           oct_voxel_xyz->distance() * a * b * c;

    const bool resul_valid = (oct_voxel->weight() > 0) && (oct_voxel_x->weight() > 0) && (oct_voxel_y->weight() > 0) &&
                             (oct_voxel_z->weight() > 0) && (oct_voxel_xy->weight() > 0) && (oct_voxel_xz->weight() > 0) &&
                             (oct_voxel_yz->weight() > 0) && (oct_voxel_xyz->weight() > 0);
    return resul_valid;
}

bool TSDFVolume::get_SDF_value(const float3& pt, r32& dist) const {
    if (cfg.use_trilinear_interpolation)
        return interpolate_trilinearly(pt, dist);
    
    const OctreeNode::Ptr voxel = octree->query_voxel(pt);
    if (voxel != nullptr && voxel->weight() > 0) {
        dist = voxel->distance();
        return true;
    }   

    dist = std::numeric_limits<r32>::quiet_NaN();

    return false;
}

bool TSDFVolume::fxn(const float3& pt, r32& value) const {
    std::vector<OctreeNode::Ptr> voxels;
    std::vector<float3> cts;
    if (!get_neighbors(pt, voxels, cts))
        return false;
    r32 d = 0;
    const r32 ratio = cfg.voxel_size / cfg.resolution_size;
    for (const auto& voxel : voxels) {
        const float3 center = voxel->center();
        d += (ratio - std::abs(pt.x - center.x)) * (ratio - std::abs(pt.y - center.y)) * (ratio - std::abs(pt.z - center.z));
    }
    value = d / (ratio*ratio*ratio);
    return true;
}

bool TSDFVolume::gradient(const float3& pt, float3& grad) const {
    std::vector<OctreeNode::Ptr> voxels;
    std::vector<float3> cts;
    if (!get_neighbors(pt, voxels, cts))
        return false;

    float3 grad_sum = make_float3(0, 0, 0);
    const r32 ratio = cfg.voxel_size / cfg.resolution_size;
    for (const auto& voxel : voxels) {
        const float3 vc = voxel->center();
        grad_sum.x += -std::copysign(1.0, pt.x - vc.x) * (ratio - std::abs(pt.y - vc.y)) * (ratio - std::abs(pt.z - vc.z)) * voxel->distance();
        grad_sum.y +=  (ratio - std::abs(pt.x - vc.x)) * -std::copysign(1.0, pt.y - vc.y) * (ratio - std::abs(pt.z - vc.z)) * voxel->distance();
        grad_sum.z +=  (ratio - std::abs(pt.x - vc.x)) * (ratio - std::abs(pt.y - vc.y)) * -std::copysign(1.0, pt.z - vc.z) * voxel->distance();
    }
    const r32 ratio3 = ratio * ratio * ratio;
    grad.x = grad_sum.x / ratio3;
    grad.y = grad_sum.y / ratio3;
    grad.z = grad_sum.z / ratio3;
    return true;
}


bool TSDFVolume::get_neighbors(const float3& query_pt, 
                               std::vector<OctreeNode::Ptr>& nodes,
                               std::vector<float3>& centers) const {
    int3 vidx = make_int3(0, 0, 0);
    if (!get_voxel_index(query_pt, vidx))
        return false;
    
    const float3 vcenter = get_voxel_center(vidx);    
    if (query_pt.x < vcenter.x) 
        vidx.x -= 1;
    if (query_pt.y < vcenter.y)
        vidx.y -= 1;
    if (query_pt.z < vcenter.z) 
        vidx.z -= 1;

    const i32 max_resol = cfg.resolution_size - 1;
    if (vidx.x < 0 || vidx.x >= max_resol || vidx.y < 0 || vidx.y >= max_resol || vidx.z < 0 || vidx.z >= max_resol)
        return false;

    for (i32 dx = 0; dx <= 1; dx++)
    {
        for (i32 dy = 0; dy <= 1; dy++)
        {
            for (i32 dz = 0; dz <= 1; dz++)
            {
                int3 dquery_voxel_ct = make_int3(vidx.x + dx, vidx.y + dy, vidx.z + dz);
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

float3 TSDFVolume::get_voxel_center(const int3& pt_idx) const {
    const r32 x_shift = cfg.voxel_size * 0.5;
    const r32 y_shift = cfg.voxel_size * 0.5;
    const r32 z_shift = cfg.voxel_size * 0.5;
    const r32 x_coord = pt_idx.x * cfg.voxel_size / cfg.resolution_size - x_shift;
    const r32 y_coord = pt_idx.y * cfg.voxel_size / cfg.resolution_size - y_shift;
    const r32 z_coord = pt_idx.z * cfg.voxel_size / cfg.resolution_size - z_shift;

    return make_float3(x_coord, y_coord, z_coord);
}   


bool TSDFVolume::get_voxel_index(const float3 pt, int3& ids) const {
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