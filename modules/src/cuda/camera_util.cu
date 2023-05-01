#include "camera_util.cuh"


namespace cuphoto {


__device__ int3 camera_to_pixel(const float3 cam_pt, const float4 camera_k) {
    int3 px_coord = make_int3(0, 0, 0);
    if (cam_pt.z == 0.)
        return px_coord;
    r32 fx = camera_k.x;
    r32 fy = camera_k.y;
    r32 cx = camera_k.z;
    r32 cy = camera_k.w;
    px_coord.x = static_cast<i32>((cam_pt.x * fx) / cam_pt.z + cx);
    px_coord.y = static_cast<i32>((cam_pt.y * fy) / cam_pt.z + cy);
    px_coord.z = 1;
    return px_coord;
}

__device__ float3 pixel_to_camera(const int3 px_pt, const float4 camera_k, const r32 depth) {
    float3 camera_coord = make_float3(0.0, 0.0, 0.0);
    r32 fx = camera_k.x;
    r32 fy = camera_k.y;
    r32 cx = camera_k.z;
    r32 cy = camera_k.w;
    camera_coord.x = (r32(px_pt.x) - cx) * depth / fx;
    camera_coord.y = (r32(px_pt.y) - cy) * depth / fy;
    camera_coord.z = depth;
    return camera_coord;
}


__device__ float3 world_to_camera(const float3 world_pt, const SE3<r32>& camera_pose) {
    float3 cam_pt = camera_pose * world_pt;
    return cam_pt;
}

__device__ float3 camera_to_world(const float3 cam_pt, const SE3<r32>& camera_pose_inv) {
    float3 world_pt = camera_pose_inv * cam_pt;
    return world_pt;
}

__device__ int3 world_to_pixel(const float3 world_pt, const SE3<r32>& camera_pose, const float4 camera_k) {
    int3 px_coord = make_int3(0, 0, 0);
    if (world_pt.z == 0.)
        return px_coord;
    float3 cam_pt = world_to_camera(world_pt, camera_pose);
    px_coord = camera_to_pixel(cam_pt, camera_k);
    return px_coord;
}

__device__ float3 pixel_to_world(const int3 px_pt, const SE3<r32>& camera_pose_inv, 
                                 const float4 camera_k, const r32 depth) {
    float3 cam_pt = pixel_to_camera(px_pt, camera_k, depth);
    float3 world_pt = camera_to_world(cam_pt, camera_pose_inv);
    return world_pt;
}

}