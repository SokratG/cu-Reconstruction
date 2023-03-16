#include "point_cloud_dataset.hpp"
#include "point_cloud_types.hpp"
#include "point_cloud_utility.hpp"
#include "cp_exception.hpp"

#include <pcl/io/ply_io.h>

#include <glog/logging.h>
#include <array>

namespace cuphoto {

PointCloudDataset::PointCloudDataset(const Config& cfg) {
    _data_path = cfg.get<std::string>("dataset.path");
    color_mode = static_cast<bool>(cfg.get<i32>("dataset.color.mode", 0));
    voxel_fiter_resolution = cfg.get<r32>("dataset.pcl.voxel_filter.resolution", 0.0001);
    set_data_directory(_data_path);
}


POINTCLOUDPtr PointCloudDataset::get_next() {
    if (current_file == files.end()) {
        throw CuPhotoException("All files in directory was traversed! Use reset for iterate again.");
    }

    std::array<r64, 9> K{1, 0, 0, 0, 1, 0, 0, 1};
    cudaPointCloud::Ptr cpc;
    if (color_mode)
        cpc = handle_pc_rgb(*current_file, K);
    else
        cpc = handle_pc(*current_file, K);

    std::advance(current_file, 1);

    return cpc;
}

POINTCLOUDPtr PointCloudDataset::handle_pc_rgb(const std::string& data_path, const std::array<r64, 9>& K) {
    PointCloudCPtr pcl_pc(new PointCloudC);
    const auto res_read = pcl::io::loadPLYFile<PointTC>(data_path, *pcl_pc);
    if (res_read == -1) {
        throw CuPhotoException("Can't read ply file.");
    }
    VoxelFilterConfig vfc;
    vfc.resolution = voxel_fiter_resolution;
    pcl_pc = voxel_filter_pc(pcl_pc, vfc);
    return pcl_to_cuda_pc(pcl_pc, K);
}

POINTCLOUDPtr PointCloudDataset::handle_pc(const std::string& data_path, const std::array<r64, 9>& K) {
    PointCloudPtr pcl_pc(new PointCloud);
    const auto res_read = pcl::io::loadPLYFile<PointT>(data_path, *pcl_pc);
    if (res_read == -1) {
        throw CuPhotoException("Can't read ply file.");
    }
    VoxelFilterConfig vfc;
    vfc.resolution = voxel_fiter_resolution;
    pcl_pc = voxel_filter_pc(pcl_pc, vfc);
    return pcl_to_cuda_pc(pcl_pc, K);
}

}