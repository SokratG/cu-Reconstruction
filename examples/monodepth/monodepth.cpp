#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "rgb_dataset.hpp"
#include "depth_estimator_mono.hpp"

#include <opencv2/imgproc.hpp>

DEFINE_string(config_path, "./data/monodepth_parameters.yaml", "file path with parameters");
DEFINE_string(store_depth_path, "./depth.png", "file path for store depth image");


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    cuphoto::RGBDataset rgb_dataset(cfg.get<std::string>("dataset.path"));
    const auto image = rgb_dataset.get_next();
    cv::cuda::GpuMat input_frame;
    image.convertTo(input_frame, CV_32FC3);
    
    try {
        cuphoto::DepthEstimatorMono de_mono(cfg);
        const auto gpu_depth = de_mono.process(input_frame);
        
        LOG(INFO) << "Depth width: " << gpu_depth.cols;
        LOG(INFO) << "Depth height: " << gpu_depth.rows;

        const std::string filepath = FLAGS_store_depth_path;

        cv::Mat cpu_depth;
        gpu_depth.download(cpu_depth);
        cv::normalize(cpu_depth, cpu_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(cpu_depth, cpu_depth, cv::COLORMAP_MAGMA);
        
        rgb_dataset.store_image(filepath, cpu_depth);
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