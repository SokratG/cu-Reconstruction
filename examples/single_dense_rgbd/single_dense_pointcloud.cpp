#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "camera.hpp"
#include "rgb_dataset.hpp"
#include "single_view_scene_rgbd.hpp"


DEFINE_string(config_path, "./data/single_rgbd_parameters.yaml", "file path with parameters");


int main(int argc, char* argv[]) {
     google::ParseCommandLineFlags(&argc, &argv, true);

    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;


    cuphoto::Camera::Ptr camera_rgb = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                        cfg.get<float>("camera.fy"), 
                                                                        cfg.get<float>("camera.cx"),
                                                                        cfg.get<float>("camera.cy"));
    cuphoto::Camera::Ptr camera_depth = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                          cfg.get<float>("camera.fy"), 
                                                                          cfg.get<float>("camera.cx"),
                                                                          cfg.get<float>("camera.cy"));

    cuphoto::RGBDataset rgb_dataset(cfg.get<std::string>("dataset.path"));
    
    cuphoto::SingleViewSceneRGBD svsd(camera_rgb, camera_depth);

    const std::string store_path = cfg.get<std::string>("point_cloud_store.path");
    
    // TODO
    
    return EXIT_SUCCESS;
}