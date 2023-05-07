#ifndef CUPHOTO_LIB_TSDF_SURFACE_RECONSTRUCTION_HPP
#define CUPHOTO_LIB_TSDF_SURFACE_RECONSTRUCTION_HPP

#include "surface_reconstruction.hpp"

namespace cuphoto {

struct TSDFVolumeConfig;

class TSDFSurface : public SurfaceReconstruction {
public:
    TSDFSurface(const Config& cfg);

    void reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) override;
    void reconstruct_surface(const cv::cuda::GpuMat& color,
                             const cv::cuda::GpuMat& depth);

    void set_global_transformation(const SE3& transform);
    SE3 get_global_transformation() const;
private:
    TSDFVolumeConfig build_cfg() const;
private:
    SE3 global_transform;
    r64 normals_radius_search;
    i32 k_nn;

    Vec3 voxel_grid_size;
    Vec3 physical_size;
    Vec3 global_offset;
    r32 max_weight;
    r32 focal_x;
    r32 focal_y;
    r32 principal_x;
    r32 principal_y;
    ui32 camera_width;
    ui32 camera_height;
};

}

#endif