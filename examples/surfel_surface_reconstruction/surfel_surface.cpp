#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "rgb_dataset.hpp"
#include "depth_estimator_mono.hpp"
#include "single_view_scene_rgbd.hpp"
#include "surfel_surface_reconstruction.hpp"


DEFINE_string(config_path, "./data/surfel_surface_parameters.yaml", "file path with parameters");


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    cuphoto::RGBDataset rgb_dataset(cfg.get<std::string>("dataset.path"));
    const int img_idx = cfg.get<int>("images.index");
    rgb_dataset.step_on(img_idx);
    const auto image = rgb_dataset.get_next();
    cv::cuda::GpuMat input_frame;
    image.convertTo(input_frame, CV_32FC3);

    cuphoto::Camera::Ptr camera_rgb = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                        cfg.get<float>("camera.fy"), 
                                                                        cfg.get<float>("camera.cx"),
                                                                        cfg.get<float>("camera.cy"));
    cuphoto::Camera::Ptr camera_depth = std::make_shared<cuphoto::Camera>(cfg.get<float>("camera.fx"),
                                                                          cfg.get<float>("camera.fy"), 
                                                                          cfg.get<float>("camera.cx"),
                                                                          cfg.get<float>("camera.cy"));
    const std::string store_path = cfg.get<std::string>("mesh_store.path");

    try {
        cuphoto::DepthEstimatorMono de_mono(cfg);
        const auto gpu_depth = de_mono.process(input_frame);

        cuphoto::SingleViewSceneRGBD svs(camera_rgb, camera_depth);

        svs.reconstruct_scene(image, gpu_depth, cfg);

        const auto point_cloud = svs.get_point_cloud();

        cuphoto::SurfelSurface ss(cfg);
        ss.reconstruct_surface(point_cloud);

        const auto mesh = ss.get_mesh();

        mesh.store_to_ply(store_path);

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