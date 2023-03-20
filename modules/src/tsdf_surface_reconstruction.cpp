#include "tsdf_surface_reconstruction.hpp"
#include "point_cloud_utility.hpp"

#include "cp_exception.hpp"
#include "cuda/tsdf_volume.cuh"

#include <glog/logging.h>

namespace cuphoto {

TSDFSurface::TSDFSurface(const Config& cfg) {
    // TODO
    
}

void TSDFSurface::reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) {
    // TODO
    TSDFVolume tsdf_volume;

    tsdf_volume.integrate();
    
}

}