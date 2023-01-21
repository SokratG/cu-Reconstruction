#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "camera.hpp"
#include "rgbd_dataset.hpp"
#include "rgb_dataset.hpp"
#include "sfm.hpp"


// DEFINE_string(dir_path, "./data/dataset/rgbd/scene01", "dataset file path");

DEFINE_string(dir_path, "./data/dataset/monocular/figure", "dataset file path");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    // google::InitGoogleLogging(argv[0]);

    // color and depth intrinsic Kinect v2.
    // cv::Mat K_color = (cv::Mat_<r64>(3, 3) << 527.01, 0, 320.0, 0, 527.01, 240.0, 0, 0, 1);
    // cuphoto::Camera::Ptr kinect_rgb = std::make_shared<cuphoto::Camera>(1058.26, 1058.26, 320.0,  240.0);
    // cuphoto::Camera kinect_depth = std::make_shared<cuphoto::Camera>(391.54, 391.54, 265.94, 218.74);
    cuphoto::Camera::Ptr kinect_rgb = std::make_shared<cuphoto::Camera>(527.01, 527.01, 320.0,  240.0);

    // cuphoto::RGBDDataset rgbd_dataset(FLAGS_dir_path);
    cuphoto::RGBDataset rgb_dataset(FLAGS_dir_path);
    cuphoto::Sfm sfm(kinect_rgb);
    for (auto i = 0; i < rgb_dataset.num_files(); ++i) {
        // auto [rgb, depth] = rgbd_dataset.get_next();
        auto rgb = rgb_dataset.get_next();
        sfm.add_frame(rgb);
    }

    sfm.run_pipeline();
    sfm.store_to_ply("some.ply", 40.0);

    return EXIT_SUCCESS;
}