#ifndef CUPHOTO_LIB_TSDF_SURFACE_RECONSTRUCTION_HPP
#define CUPHOTO_LIB_TSDF_SURFACE_RECONSTRUCTION_HPP

#include "surface_reconstruction.hpp"

namespace cuphoto {


class TSDFSurface : public SurfaceReconstruction {
public:
    TSDFSurface(const Config& cfg);

    void reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) override;

    void set_global_transformation(const SE3& transform);
    SE3 get_global_transformation() const;
private:
    SE3 global_transform;
    r64 normals_radius_search;
    i32 k_nn;
    r32 voxel_size;
    i32 resolution_size;
    r32 max_dist_p;
    r32 max_dist_n;
    r32 max_weight;
    r32 min_sensor_dist;
    r32 max_sensor_dist;
    r32 focal_x;
    r32 focal_y;
    r32 principal_x;
    r32 principal_y;
    i32 weight_type;
    r32 max_cell_size;
    i32 num_rand_split;
    ui32 pool_size;
    i32 n_level_split;
    Vec3 center_octree;
    bool use_trilinear_interpolation;
};

}

#endif