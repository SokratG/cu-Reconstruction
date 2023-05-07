#ifndef CUPHOTO_LIB_TSDF_VOLUME_CUH
#define CUPHOTO_LIB_TSDF_VOLUME_CUH

#include "types.cuh"
#include "cuda_point_cloud.cuh"
#include <curand_kernel.h>

#include <memory>

namespace cuphoto {

enum WEIGHT_TYPE {
    WEIGHT_BY_DEPTH = 0,
    WEIGHT_BY_VARIANCE,
};

struct TSDFVolumeConfig {
    dim3 voxel_grid_size;
    float3 physical_size; // in physical units
    float3 global_offset; // grid_offset
    r32 max_weight;
    r32 focal_x;
    r32 focal_y;
    r32 cx;
    r32 cy;
    ui32 camera_width;
    ui32 camera_height;
    r32 max_cell_size = 0.0;
};

class TSDFVolume {
public:
    struct TransformationVoxel {
        using Ptr = TransformationVoxel*;

        float3 translation;
        float4 quat_rotation;
    } __attribute__((packed));
public:
    using Ptr = std::shared_ptr<TSDFVolume>;

    TSDFVolume(const TSDFVolumeConfig& tsdf_cfg);
    ~TSDFVolume();

    bool integrate(const cudaPointCloud::Ptr cuda_pc,
                   cv::cuda::PtrStepSzf depth,
                   const std::array<r64, 7>& camera_pose);

    bool integrate(const cv::cuda::PtrStepSzb color,
                   const cv::cuda::PtrStepSzf depth,
                   const std::array<r64, 7>& camera_pose);

    bool apply_isosurface_transformation(cudaPointCloud::Vertex::Ptr vertices,
                                         const ui32 num_pts, 
                                         const std::array<r64, 7>& camera_pose);

    r32* voxel_distances_data() {
        return voxel_distances;
    }

    r32* voxel_distances_data() const {
        return voxel_distances;
    }

    r32* voxel_weights_data() {
        return voxel_weights;
    }

    dim3 voxel_grid_size() const {
        return cfg.voxel_grid_size;
    }

    float3 global_offset() const {
        return cfg.global_offset;
    }

    float3 scale_voxel_size() const {
        return voxel_size;
    }

    float3 voxel_phys_size() const {
        return cfg.physical_size;
    }
    

    TransformationVoxel::Ptr transformation_voxels_data() {
        return t_voxels;
    }

    TransformationVoxel::Ptr transformation_voxels_data() const {
        return t_voxels;
    }
private:
    void free_data();
    void setup_data();
private:
    TSDFVolumeConfig cfg;
    float3 voxel_size;
    r32 max_truncated_dist;

    r32* voxel_distances = nullptr;
    r32* voxel_weights = nullptr;
    TransformationVoxel::Ptr t_voxels = nullptr;
    curandState* custate = nullptr;
};

}


#endif