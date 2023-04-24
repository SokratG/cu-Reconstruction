#ifndef CUPHOTO_LIB_FRUSTRUM_CUH
#define CUPHOTO_LIB_FRUSTRUM_CUH

#include "types.cuh"
#include "object_pool_allocator.cuh"
#include "se3.cuh"


namespace cuphoto {

class OctreeNode;

struct Plane {
    using Ptr = Plane*;

    float3 normal = make_float3(0.0, 0.0, 0.0);
    r32 dist = 0.0;

    __host__ __device__ Plane() {};

	__host__ __device__ Plane(const float3& pos, const float3& norm)
		                     : normal(normalize(norm)), dist(dot(normal, pos)) {}

    __host__ __device__ float sd_to_plane(const float3& pt);

    __host__ __device__ bool is_on_forward_plane(const OctreeNode* voxel);

} __attribute__((packed));


class Frustrum : CudaManaged {
public:
    using Ptr = Frustrum*;

    __host__ static Frustrum* create_frustrum(const SE3<r32>& global_pose, const r32 aspect, const r32 fovY, const r32 zNear, const r32 zFar);

    __host__ static void free_frustrum(Frustrum* frustrum);

    __host__ __device__ void collision_frustrum(OctreeNode* voxel);

protected:
    Frustrum() {}

    Plane top_face;
    Plane bottom_face;

    Plane left_face;
    Plane right_face;

    Plane near_face;
    Plane far_face;
};

}

#endif