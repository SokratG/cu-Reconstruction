#include "tsdf_surface_reconstruction.hpp"
#include "point_cloud_utility.hpp"
#include "mesh.hpp"

#include "cp_exception.hpp"
#include "cuda/tsdf_volume.cuh"
#include "cuda/marching_cubes.cuh"

#include <glog/logging.h>

namespace cuphoto {

TSDFVolumeConfig TSDFSurface::build_cfg() const {
    TSDFVolumeConfig tsdf_cfg;

    tsdf_cfg.voxel_grid_size.x = voxel_grid_size.x();
    tsdf_cfg.voxel_grid_size.y = voxel_grid_size.y();
    tsdf_cfg.voxel_grid_size.z = voxel_grid_size.z();

    tsdf_cfg.physical_size.x = physical_size.x();
    tsdf_cfg.physical_size.y = physical_size.y();
    tsdf_cfg.physical_size.z = physical_size.z();

    tsdf_cfg.global_offset.x = global_offset.x();
    tsdf_cfg.global_offset.y = global_offset.y();
    tsdf_cfg.global_offset.z = global_offset.z();

    tsdf_cfg.max_weight = max_weight;

    tsdf_cfg.focal_x = focal_x;
    tsdf_cfg.focal_y = focal_y;
    tsdf_cfg.cx = principal_x;
    tsdf_cfg.cy = principal_y;

    tsdf_cfg.camera_width = camera_width;
    tsdf_cfg.camera_height = camera_height;
    return tsdf_cfg;
}


TSDFSurface::TSDFSurface(const Config& cfg) {
    normals_radius_search = cfg.get<r64>("pcl.normals.radius_search");
    k_nn = cfg.get<i32>("pcl.normals.k_nn", 0);

    voxel_grid_size.x() = cfg.get<r32>("tsdf.volume.voxel_grid_size.x");
    voxel_grid_size.y() = cfg.get<r32>("tsdf.volume.voxel_grid_size.y");
    voxel_grid_size.z() = cfg.get<r32>("tsdf.volume.voxel_grid_size.z");

    physical_size.x() = cfg.get<r32>("tsdf.volume.physical_size.x");
    physical_size.y() = cfg.get<r32>("tsdf.volume.physical_size.y");
    physical_size.z() = cfg.get<r32>("tsdf.volume.physical_size.z");

    max_weight = cfg.get<r32>("tsdf.volume.max_weight");

    global_offset.x() = cfg.get<r32>("tsdf.volume.global_offset.x", 0.);
    global_offset.y() = cfg.get<r32>("tsdf.volume.global_offset.y", 0.);
    global_offset.z() = cfg.get<r32>("tsdf.volume.global_offset.z", 0.);

    focal_x = cfg.get<r32>("camera.fx", 1.0);
    focal_y = cfg.get<r32>("camera.fy", 1.0);
    principal_x = cfg.get<r32>("camera.cx", 0);
    principal_y = cfg.get<r32>("camera.cy", 0);
    camera_width = cfg.get<r32>("camera.width");
    camera_height = cfg.get<r32>("camera.height");

    const r32 roll_degree = cfg.get<r32>("tsdf.volume.camera.pose.euler.roll", 0.);
    const r32 yaw_degree = cfg.get<r32>("tsdf.volume.camera.pose.euler.yaw", 0.);
    const r32 pitch_degree = cfg.get<r32>("tsdf.volume.camera.pose.euler.pitch", 0.);

    constexpr r32 half_c = M_PI / 180.;
    Eigen::AngleAxisd roll_angle(roll_degree * half_c, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd yaw_angle(yaw_degree * half_c, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitch_angle(pitch_degree * half_c, Eigen::Vector3d::UnitX());
    Quat qrot = roll_angle * yaw_angle * pitch_angle;

    Vec3 t(cfg.get<r32>("tsdf.volume.camera.pose.translation.x", 0.),
           cfg.get<r32>("tsdf.volume.camera.pose.translation.y", 0.), 
           cfg.get<r32>("tsdf.volume.camera.pose.translation.z", 0.));

    SE3 global_camera_pose(qrot, t);
    set_global_transformation(global_camera_pose);
}


void TSDFSurface::set_global_transformation(const SE3& transform) {
    global_transform = transform;
}


SE3 TSDFSurface::get_global_transformation() const {
    return global_transform;
}

void TSDFSurface::reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) {
    // TODO
    TSDFVolumeConfig tsdf_cfg = build_cfg();

    TSDFVolume tsdf_volume(tsdf_cfg);

    const auto cuda_normals_pc = compute_normals_pc(cuda_pc, normals_radius_search, k_nn);

    Mat3 R = global_transform.rotationMatrix();    
    Vec3 t = global_transform.translation();
    Quat q(R);
    std::array<r64, 7> global_pose {q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z()};

    cv::cuda::GpuMat gpu_depth(cv::Size(camera_width, camera_height), CV_32F);
    cv::cuda::GpuMat gpu_color(cv::Size(camera_width, camera_height), CV_8UC3);
    gpu_color.step = camera_width * sizeof(uchar3);
    if (!cuda_pc->project_to_color_depth(gpu_color, gpu_depth, global_pose)) {
        return;
    }

    // tsdf_volume.integrate(gpu_color, gpu_depth, global_pose);
    
    // std::vector<float3> verts;
    // std::vector<int3> triangles;
    // generate_triangular_surface(tsdf_volume, verts, triangles);

}

void TSDFSurface::reconstruct_surface(const cv::cuda::GpuMat& color, const cv::cuda::GpuMat& depth) {
    TSDFVolumeConfig tsdf_cfg = build_cfg();

    TSDFVolume tsdf_volume(tsdf_cfg);

    Mat3 R = global_transform.rotationMatrix();    
    Vec3 t = global_transform.translation();
    Quat q(R);
    std::array<r64, 7> global_pose {q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z()};

    // gpu_color.step = camera_width * sizeof(uchar3);
    
    tsdf_volume.integrate(color, depth, global_pose);
    
    std::vector<float3> verts;
    std::vector<int3> triangles;
    generate_triangular_surface(tsdf_volume, verts, triangles);

    std::vector<Mesh::Point> mesh_verts(verts.size());
    std::vector<Mesh::Triangle> mesh_triangles(triangles.size());

    for (i32 idx = 0; idx < verts.size(); ++idx) {
        mesh_verts[idx] = Mesh::Point(verts[idx].x, verts[idx].y, verts[idx].z);
    }
    
    for (i32 idx = 0; idx < triangles.size(); ++idx) {
        mesh_triangles[idx] = Mesh::Triangle(triangles[idx].x, triangles[idx].y, triangles[idx].z);
    }

    this->surface_mesh.vertices(mesh_verts);
    this->surface_mesh.polygons(mesh_triangles);
}

}