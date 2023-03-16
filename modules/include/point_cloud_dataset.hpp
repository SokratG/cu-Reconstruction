#ifndef POINT_CLOUD_DATASET_HPP
#define POINT_CLOUD_DATASET_HPP

#include "cuda/cuda_point_cloud.cuh"
#include "config.hpp"
#include "utils.hpp"

#include <string>
#include <set>

namespace cuphoto {

using POINTCLOUD = cudaPointCloud;
using POINTCLOUDPtr = cudaPointCloud::Ptr;

class PointCloudDataset {
public:
    PointCloudDataset(const Config& cfg);

    virtual i32 num_files() const {
        return files.size();
    };

    void set_data_directory(const std::string& data_path) {
        this->data_path(data_path);
        files = files_directory(data_path, extensions);
        current_file = files.begin();
    }

    std::string data_path() const {
        return _data_path;
    }

    void step_on(const i32 adv_step) {
        if (adv_step >= files.size())
            return;
        std::advance(current_file, adv_step);
    }

    virtual POINTCLOUDPtr get_next();

protected:
    void data_path(const std::string& data_path) {
        _data_path = data_path;
    }

    virtual POINTCLOUDPtr handle_pc_rgb(const std::string& data_path, const std::array<r64, 9>& K);
    virtual POINTCLOUDPtr handle_pc(const std::string& data_path, const std::array<r64, 9>& K);

private:
    const std::set<std::string> extensions {".ply"};
    std::string _data_path;
    std::set<std::string> files;
    std::set<std::string>::iterator current_file;
    r32 voxel_fiter_resolution;
    bool color_mode;
};

}

#endif