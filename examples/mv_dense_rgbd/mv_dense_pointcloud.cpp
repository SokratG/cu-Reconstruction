#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "camera.hpp"
#include "rgbd_dataset.hpp"
#include "multi_view_scene_rgbd.hpp"


DEFINE_string(config_path, "./data/mv_rgbd_parameters.yaml", "file path with parameters");


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    cuphoto::Camera::Ptr kinect_rgb = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                        cfg.get<float>("camera.fy"), 
                                                                        cfg.get<float>("camera.cx"),
                                                                        cfg.get<float>("camera.cy"));
    cuphoto::Camera::Ptr kinect_depth = std::make_shared<cuphoto::Camera>(cfg.get<float>("depth.fx"),
                                                                          cfg.get<float>("depth.fy"), 
                                                                          cfg.get<float>("depth.cx"),
                                                                          cfg.get<float>("depth.cy"));

    cuphoto::RGBDDataset rgbd_dataset(cfg.get<std::string>("dataset.path"), cfg.get<float>("depth_scale"));
    const int N = cfg.get<int>("num_images");
    cuphoto::MultiViewSceneRGBD mvs(kinect_rgb, kinect_depth);
    for (auto i = 0; i < N; ++i) {
        auto [rgb, depth] = rgbd_dataset.get_next();
        mvs.add_frame(rgb, depth);
    }
    const std::string store_path = cfg.get<std::string>("point_cloud_store.path");

    try {
        mvs.reconstruct_scene(cfg);
        mvs.store_to_ply(store_path);
    } catch(cuphoto::CuPhotoException& photo_ex) {
        LOG(ERROR) << photo_ex.what();
        return EXIT_FAILURE;
    } catch (std::exception& ex) {
        LOG(ERROR) << ex.what();
        return EXIT_FAILURE;
    } catch(...) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}