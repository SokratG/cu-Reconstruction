#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "camera.hpp"
#include "stereo_dataset.hpp"
#include "depth_estimator_stereo.hpp"
#include "single_view_scene_rgbd.hpp"

#include <opencv2/imgproc.hpp>


DEFINE_string(config_path, "./data/stereodepth_parameters.yaml", "file path with parameters");



int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    cuphoto::Camera::Ptr camera_rgb = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                        cfg.get<float>("camera.fy"), 
                                                                        cfg.get<float>("camera.cx"),
                                                                        cfg.get<float>("camera.cy"));
    
    const float baseline =  cfg.get<float>("camera.stereo.baseline");
    const std::string store_path_pc = cfg.get<std::string>("point_cloud_store.path");
    const std::string store_path_stereo_depth = cfg.get<std::string>("stereo_depth.path");

    cuphoto::StereoDataset stereo_dataset(cfg.get<std::string>("dataset.path"));

    auto [left_img, right_img] = stereo_dataset.get_next();

    cuphoto::DepthEstimatorStereoSGM desSGM(camera_rgb, camera_rgb, cfg);

    const auto gpu_depth = desSGM.estimate_depth(left_img, right_img, baseline);

    LOG(INFO) << "Depth width: " << gpu_depth.cols;
    LOG(INFO) << "Depth height: " << gpu_depth.rows;

    cv::Mat cpu_depth;
    gpu_depth.download(cpu_depth);
    cv::normalize(cpu_depth, cpu_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(cpu_depth, cpu_depth, cv::COLORMAP_MAGMA);
    
    stereo_dataset.store_image(store_path_stereo_depth, cpu_depth);

    cuphoto::SingleViewSceneRGBD svs(camera_rgb, camera_rgb);

    svs.reconstruct_scene(left_img, gpu_depth, cfg);

    svs.store_to_ply(store_path_pc);

    return EXIT_SUCCESS;
}