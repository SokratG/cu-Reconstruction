#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "rgbd_dataset.hpp"

DEFINE_string(dir_path, "./data/dataset/rgbd/scene01", "dataset file path");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    // google::InitGoogleLogging(argv[0]);

    curec::RGBDDataset rgbd_dataset(FLAGS_dir_path);

    for (auto i = 0; i < 6; ++i) {
        auto [rgb, depth] = rgbd_dataset.get_next();
        if (rgb.empty()) {
            LOG(INFO) << "HELLO";
        }
    }

    return EXIT_SUCCESS;
}