#ifndef CUPHOTO_LIB_SURFACE_RECONSTRUCTION_HPP
#define CUPHOTO_LIB_SURFACE_RECONSTRUCTION_HPP

#include "cuda/cuda_point_cloud.cuh"
#include "mesh.hpp"

#include "config.hpp"

namespace cuphoto {

class SurfaceReconstruction {
public:
    SurfaceReconstruction() = default;

    virtual void reconstruct_surface(const cudaPointCloud::Ptr cuda_pc, const Config& cfg) = 0;

    Mesh get_mesh() const {
        return surface_mesh;
    }
private:
    Mesh surface_mesh;
};


};

#endif