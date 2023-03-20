#include "tsdf_surface_reconstruction.hpp"
#include "point_cloud_utility.hpp"

#include "cp_exception.hpp"
#include "cuda/tsdf_volume.cuh"

#include <glog/logging.h>

namespace cuphoto {

TSDFSurface::TSDFSurface(const Config& cfg) {
    // TODO
    normals_radius_search = cfg.get<r64>("pcl.normals.radius_search");
    k_nn = cfg.get<i32>("pcl.normals.k_nn", 0);
}

void TSDFSurface::reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) {
    // TODO
    TSDFVolume tsdf_volume;

    const auto cuda_normals_pc = compute_normals_pc(cuda_pc, normals_radius_search, k_nn);

    tsdf_volume.integrate();
    
}

}