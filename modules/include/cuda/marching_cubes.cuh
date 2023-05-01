#ifndef CUPHOTO_LIB_MARCHING_CUBES_CUH
#define CUPHOTO_LIB_MARCHING_CUBES_CUH

#include "types.cuh"
#include "tsdf_volume.cuh"
#include <cuda_runtime.h>
#include <vector>

namespace cuphoto {


// 
void generate_triangular_surface(const TSDFVolume& tsdf_volume, 
                                 std::vector<float3>& verts, 
                                 std::vector<int3>& triangles);

}



#endif