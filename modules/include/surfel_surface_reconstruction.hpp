#ifndef CUPHOTO_LIB_SURFEL_SURFACE_RECONSTRUCTION_HPP
#define CUPHOTO_LIB_SURFEL_SURFACE_RECONSTRUCTION_HPP


#include "surface_reconstruction.hpp"

namespace cuphoto {

class SurfelSurface : public SurfaceReconstruction {
public:
    void reconstruct_surface(const cudaPointCloud::Ptr cuda_pc, const Config& cfg) override;


};

}

#endif