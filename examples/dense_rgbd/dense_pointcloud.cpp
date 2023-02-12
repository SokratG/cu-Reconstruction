#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "camera.hpp"
#include "rgbd_dataset.hpp"
#include "multi_view_scene_rgbd.hpp"


DEFINE_string(dir_path, "./data/dataset/rgbd/scene01", "dataset file path");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    cuphoto::Camera::Ptr kinect_rgb = std::make_shared<cuphoto::Camera>(527.01, 527.01, 320.0,  240.0);
    cuphoto::Camera::Ptr kinect_depth = std::make_shared<cuphoto::Camera>(391.54, 391.54, 265.94, 218.74);

    cuphoto::RGBDDataset rgbd_dataset(FLAGS_dir_path, 0.001);
    cuphoto::MultiViewSceneRGBD mvs(kinect_rgb, kinect_depth);
    for (auto i = 0; i < 5; ++i) {
        auto [rgb, depth] = rgbd_dataset.get_next();
        mvs.add_frame(rgb, depth);
    }

    mvs.reconstruct_scene();
    mvs.store_to_ply("dense_rgbd_point_cloud.ply");

    return EXIT_SUCCESS;
}