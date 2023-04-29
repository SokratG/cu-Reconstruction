#ifndef CUPHOTO_LIB_TSDF_VOLUME_CUH
#define CUPHOTO_LIB_TSDF_VOLUME_CUH

#include "types.cuh"
#include "octree.cuh"
#include "cuda_point_cloud.cuh"
#include "util.cuh"

#include <memory>

namespace cuphoto {

enum WEIGHT_TYPE {
    WEIGHT_BY_DEPTH = 0,
    WEIGHT_BY_VARIANCE,
};

struct TSDFVolumeConfig {
    r32 voxel_size;
    i32 resolution_size;
    r32 max_dist_p;
    r32 max_dist_n;
    r32 max_weight;
    r32 min_sensor_dist;
    r32 max_sensor_dist;
    r32 focal_x;
    r32 focal_y;
    r32 cx;
    r32 cy;
    WEIGHT_TYPE weight_type = WEIGHT_TYPE::WEIGHT_BY_DEPTH;
    r32 max_cell_size;
    i32 num_rand_split = 1;
    i32 n_level_split = 0;
    float3 center_octree = make_float3(0, 0, 0);
    ui32 pool_size;
    bool use_trilinear_interpolation = true;
};

class TSDFVolume {
public:
    using Ptr = std::shared_ptr<TSDFVolume>;

    TSDFVolume(const TSDFVolumeConfig& tsdf_cfg, const i32 reserve_pool_size = 0);
    ~TSDFVolume();

    bool integrate(const cudaPointCloud::Ptr cuda_pc,
                   const std::array<r64, 7>& global_pose);

    bool get_neighbors(const float3& query_pt, std::vector<OctreeNode::Ptr>& nodes, std::vector<float3>& centers) const;
    std::vector<int3> get_occupied_voxel_indices() const;
    bool get_SDF_value(const float3& pt, r32& dist) const;
    bool fxn(const float3& pt, r32& value) const;
    bool gradient(const float3& pt, float3& grad) const;
public:
    Octree::Ptr octree;
private:
    void distance_to_voxel(const cudaPointCloud::Ptr cuda_pc);

    bool interpolate_trilinearly(const float3& pt, r32& dist) const;
    bool get_voxel_index(const float3 pt, int3& ids) const;
    float3 get_voxel_center(const int3& coord) const;
private:
    TSDFVolumeConfig cfg;
    curandState* custate = nullptr;
};

}


#endif