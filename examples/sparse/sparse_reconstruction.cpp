#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "camera.hpp"
#include "rgb_dataset.hpp"
#include "sfm.hpp"


// DEFINE_string(data_path, "./data/dataset/monocular/figure", "dataset file path");
DEFINE_string(config_path, "./data/sfm_parameters.yaml", "file path with parameters");


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    // color and depth intrinsic Kinect v2.
    // cv::Mat K_color = (cv::Mat_<r64>(3, 3) << 527.01, 0, 320.0, 0, 527.01, 240.0, 0, 0, 1);
    // cuphoto::Camera::Ptr kinect_rgb = std::make_shared<cuphoto::Camera>(1058.26, 1058.26, 320.0,  240.0);
    // cuphoto::Camera kinect_depth = std::make_shared<cuphoto::Camera>(391.54, 391.54, 265.94, 218.74);
    cuphoto::Camera::Ptr kinect_rgb = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                        cfg.get<float>("camera.fy"), 
                                                                        cfg.get<float>("camera.cx"),
                                                                        cfg.get<float>("camera.cy"));

    cuphoto::RGBDataset rgb_dataset(cfg.get<std::string>("dataset.path"));
    cuphoto::Sfm sfm(kinect_rgb);
    for (auto i = 0; i < rgb_dataset.num_files(); ++i) {
        auto rgb = rgb_dataset.get_next();
        sfm.add_frame(rgb);
    }

    sfm.run_pipeline(cfg);
    const float x_min = cfg.get<float>("sfm.range_threshold.x.min");
    const float x_max = cfg.get<float>("sfm.range_threshold.x.max");
    const float y_min = cfg.get<float>("sfm.range_threshold.y.min");
    const float y_max = cfg.get<float>("sfm.range_threshold.y.max");
    const float depth = cfg.get<float>("sfm.range_threshold.z");
    const std::string store_path = cfg.get<std::string>("point_cloud_store.path");

    sfm.store_to_ply(store_path, x_min, x_max, y_min, y_max, depth);

    return EXIT_SUCCESS;
}