#ifndef CUPHOTO_LIB_SURFEL_SURFACE_RECONSTRUCTION_HPP
#define CUPHOTO_LIB_SURFEL_SURFACE_RECONSTRUCTION_HPP


#include "surface_reconstruction.hpp"

namespace cuphoto {

struct SurfaceData;

class SurfelSurface : public SurfaceReconstruction {
public:
    SurfelSurface(const Config& cfg);

    void reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) override;
private:
    void build_surfel_point_cloud(SurfaceData& sd);
    void triangulate(SurfaceData& sd);
    void build_mesh(SurfaceData& sd);
private:
    r32 mls_radius;
    i32 polynomial_order;
    r32 triangle_search_radius;
    r32 mu;
    i32 max_nn;
    r32 max_surf_angle;
    r32 min_angle;
    r32 max_angle;
    bool normal_consistency;
};

}

#endif