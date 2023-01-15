#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "camera.hpp"
#include "rgbd_dataset.hpp"
#include "sfm.hpp"


DEFINE_string(dir_path, "./data/dataset/rgbd/scene01", "dataset file path");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    // google::InitGoogleLogging(argv[0]);

    // color and depth intrinsic Kinect v2.
    // cv::Mat K_color = (cv::Mat_<r64>(3, 3) << 527.01, 0, 320.0, 0, 527.01, 240.0, 0, 0, 1);
    // curec::Camera::Ptr kinect_rgb = std::make_shared<curec::Camera>(1058.26, 1058.26, 320.0,  240.0);
    // curec::Camera kinect_depth = std::make_shared<curec::Camera>(391.54, 391.54, 265.94, 218.74);
    curec::Camera::Ptr kinect_rgb = std::make_shared<curec::Camera>(527.01, 527.01, 320.0,  240.0);

    curec::RGBDDataset rgbd_dataset(FLAGS_dir_path);
    curec::Sfm sfm(kinect_rgb);
    for (auto i = 0; i < 6; ++i) {
        auto [rgb, depth] = rgbd_dataset.get_next();
        sfm.add_frame(rgb);
    }

    sfm.run_pipeline();
    sfm.store_to_ply("some.ply");

    return EXIT_SUCCESS;
}