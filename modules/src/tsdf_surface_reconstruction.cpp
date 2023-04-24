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

    voxel_size = cfg.get<r32>("tsdf.volume.voxel_size");
    resolution_size = cfg.get<i32>("tsdf.volume.resolution_size");
    max_dist_p = cfg.get<r32>("tsdf.volume.max_dist_p");
    max_dist_n = cfg.get<r32>("tsdf.volume.max_dist_n");
    max_weight = cfg.get<r32>("tsdf.volume.max_weight");
    min_sensor_dist = cfg.get<r32>("tsdf.volume.min_sensor_dist");
    max_sensor_dist = cfg.get<r32>("tsdf.volume.max_sensor_dist");
    
    focal_x = cfg.get<r32>("camera.fx", 1.0);
    focal_y = cfg.get<r32>("camera.fy", 1.0);
    principal_x = cfg.get<r32>("camera.cx", 0);
    principal_y = cfg.get<r32>("camera.cy", 0);
    weight_type = cfg.get<i32>("tsdf.volume.weight_type", 0);
    max_cell_size = cfg.get<r32>("tsdf.volume.max_cell_size");
    num_rand_split = cfg.get<i32>("tsdf.volume.num_rand_split", 1);
    n_level_split = cfg.get<i32>("tsdf.volume.octree.center.n_level_split", 0);

    const r32 center_x = cfg.get<r32>("tsdf.volume.octree.center.x", 0.);
    const r32 center_y = cfg.get<r32>("tsdf.volume.octree.center.y", 0.);
    const r32 center_z = cfg.get<r32>("tsdf.volume.octree.center.z", 0.);
    center_octree = Vec3(center_x, center_y, center_z);
    use_trilinear_interpolation = static_cast<bool>(cfg.get<i32>("tsdf.volume.use_trilinear_interpolation", 1));

}


void TSDFSurface::set_global_transformation(const SE3& transform) {
    global_transform = transform;
}


SE3 TSDFSurface::get_global_transformation() const {
    return global_transform;
}

void TSDFSurface::reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) {
    // TODO
    TSDFVolumeConfig tsdf_cfg;

    tsdf_cfg.voxel_size = voxel_size;
    tsdf_cfg.resolution_size = resolution_size;
    tsdf_cfg.max_dist_p = max_dist_p;
    tsdf_cfg.max_dist_n = max_dist_n;
    tsdf_cfg.max_weight = max_weight;

    tsdf_cfg.min_sensor_dist = min_sensor_dist;
    tsdf_cfg.max_sensor_dist = max_sensor_dist;
    tsdf_cfg.focal_x = focal_x;
    tsdf_cfg.focal_y = focal_y;
    tsdf_cfg.cx = principal_x;
    tsdf_cfg.cy = principal_y;

    tsdf_cfg.weight_type = static_cast<WEIGHT_TYPE>(weight_type);
    tsdf_cfg.max_cell_size= max_cell_size;
    tsdf_cfg.num_rand_split = num_rand_split;
    tsdf_cfg.n_level_split = n_level_split;
    tsdf_cfg.center_octree = make_float3(center_octree.x(), center_octree.y(), center_octree.z());
    tsdf_cfg.use_trilinear_interpolation = use_trilinear_interpolation;

    TSDFVolume tsdf_volume(tsdf_cfg);

    const auto cuda_normals_pc = compute_normals_pc(cuda_pc, normals_radius_search, k_nn);

    Mat3 R = global_transform.rotationMatrix();
    Vec3 t = global_transform.translation();
    cuphoto::Quat q(R);
    std::array<r64, 7> global_pose {q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z()};

    tsdf_volume.integrate(cuda_pc, global_pose);
    
}

}