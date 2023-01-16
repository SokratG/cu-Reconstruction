#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "camera.hpp"
#include "rgbd_dataset.hpp"

DEFINE_string(dir_path, "./data/dataset/rgbd/scene01", "dataset file path");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    curec::Camera::Ptr kinect_rgb = std::make_shared<curec::Camera>(527.01, 527.01, 320.0,  240.0);

    curec::RGBDDataset rgbd_dataset(FLAGS_dir_path);

    return EXIT_SUCCESS;
}