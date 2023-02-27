#include "single_view_scene_rgbd.hpp"

#include "point_cloud_utility.hpp"

#include <glog/logging.h>

namespace cuphoto {

SingleViewSceneRGBD::SingleViewSceneRGBD(const Camera::Ptr camera, const Camera::Ptr _depth_camera) :
                                         SingleViewScene(camera), depth_camera(_depth_camera)  {

}

void SingleViewSceneRGBD::reconstruct_scene(const cv::cuda::GpuMat rgb,  const cv::cuda::GpuMat depth, 
                                            const Config& cfg) {
    LOG(INFO) << "Build point cloud from depth";

    KeyFrame::Ptr kframe_rgb = KeyFrame::create_keyframe();
    kframe_rgb->frame(rgb);
    KeyFrame::Ptr kframe_depth = KeyFrame::create_keyframe();
    kframe_depth->frame(depth);

    const auto width = kframe_depth->frame().cols;
    const auto height = kframe_depth->frame().rows;
    const auto depth_threshold_min = cfg.get<r32>("point_cloud.depth_threshold_min", 1e-6);
    const auto depth_threshold_max = cfg.get<r32>("point_cloud.depth_threshold_max");
    StatisticalFilterConfig sfc;
    sfc.k_mean = cfg.get<i32>("pcl.statistical_filter.k_mean", 25);
    sfc.std_dev_mul_thresh = cfg.get<r32>("pcl.statistical_filter.std_dev_mul_thresh", 0.7);


    auto pcl_point_cloud = point_cloud_from_depth(kframe_rgb, kframe_depth, camera->K(),
                                                  depth_threshold_min, depth_threshold_max);
    
    pcl_point_cloud = statistical_filter_pc(pcl_point_cloud, sfc);

    VoxelFilterConfig vfc;
    vfc.resolution = cfg.get<r64>("pcl.voxel_filter.resolution", 0.03);
    pcl_point_cloud = voxel_filter_pc(pcl_point_cloud, vfc);

    std::array<r64, 9> K;
    Vec9::Map(K.data()) = Eigen::Map<Vec9>(camera->K().data(), camera->K().cols() * camera->K().rows());
    cuda_pc = cudaPointCloud::create(K, 0);

    cuda_pc = pcl_to_cuda_pc(pcl_point_cloud, K);
}


};