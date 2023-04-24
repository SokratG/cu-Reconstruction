#include "frustrum.cuh"
#include "octree.cuh"
#include "CudaUtils/cudaMath.cuh"
#include <cmath>

namespace cuphoto {

r32 Plane::sd_to_plane(const float3& pt) {
    r32 sdf = dot(normal, pt) - dist;
    return sdf;
}


bool Plane::is_on_forward_plane(const OctreeNode* voxel) {
    const r32 side_size = voxel->voxel_size();
    const r32 r = side_size * fabs(normal.x) + fabs(normal.y) + fabs(normal.z);
    return -r <= sd_to_plane(voxel->center());
}


Frustrum* Frustrum::create_frustrum(const SE3<r32>& global_pose, const r32 aspect, const r32 fovY, const r32 zNear, const r32 zFar) {
    Frustrum::Ptr frustrum = new Frustrum();
    const r32 cam2robot_rot[9]{0, 0, 1, 0, -1, 0, 1, 0, 0};
    const r32 cam2robot_t[3]{0, 0, 0};
    SE3<r32> cam2robot(cam2robot_rot, cam2robot_t);
    SE3<r32> cam_pose = global_pose * cam2robot;

    float3 view = make_float3(cam_pose(0, 0), cam_pose(1, 0), cam_pose(2, 0));
    float3 up = make_float3(cam_pose(0, 1), cam_pose(1, 1), cam_pose(2, 1));
    float3 right = make_float3(cam_pose(0, 2), cam_pose(1, 2), cam_pose(2, 2));

    float3 cam_position = make_float3(cam_pose(0, 3), cam_pose(1, 3), cam_pose(2, 3));

    const r32 halfVSide = zFar * tanf(fovY * .5f);
    const r32 halfHSide = halfVSide * aspect;
    const float3 frontMultFar = zFar * view;

    frustrum->near_face = Plane(cam_position + zNear * view, view);
    frustrum->far_face = Plane(cam_position + frontMultFar, -view);

    frustrum->right_face = Plane(cam_position, cross(frontMultFar - right * halfHSide, up));
    frustrum->left_face = Plane(cam_position, cross(up, frontMultFar + right * halfHSide));

    frustrum->top_face = Plane(cam_position, cross(right, frontMultFar - up * halfVSide));
    frustrum->bottom_face = Plane(cam_position, cross(frontMultFar + up * halfVSide, right));

    return frustrum;
}


void Frustrum::free_frustrum(Frustrum::Ptr frustrum) {
    delete frustrum;
}


void Frustrum::collision_frustrum(OctreeNode::Ptr voxel) {
    if (voxel == nullptr)
        return;
    const bool top_collision = top_face.is_on_forward_plane(voxel);
    const bool bottom_collision = bottom_face.is_on_forward_plane(voxel);
    const bool left_collision = left_face.is_on_forward_plane(voxel);
    const bool right_collision = right_face.is_on_forward_plane(voxel);
    const bool near_collision = near_face.is_on_forward_plane(voxel);
    const bool far_collision = far_face.is_on_forward_plane(voxel);
    const bool is_collide = top_collision && bottom_collision && left_collision && right_collision && near_collision && far_collision;
    if (is_collide) {
        voxel->in_view(INSIDE_VIEW);
    } else {
        voxel->in_view(OUTSIDE_VIEW);
    }
}

}

