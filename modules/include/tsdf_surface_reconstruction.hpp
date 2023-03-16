#ifndef CUPHOTO_LIB_TSDF_SURFACE_RECONSTRUCTION_HPP
#define CUPHOTO_LIB_TSDF_SURFACE_RECONSTRUCTION_HPP

#include "surface_reconstruction.hpp"

namespace cuphoto {


class TSDFSurface : public SurfaceReconstruction {
public:
    TSDFSurface(const Config& cfg);

    void reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) override;
private:
    
};

}

#endif