#ifndef CUPHOTO_LIB_POINT_CLOUD_STITCHER_HPP
#define CUPHOTO_LIB_POINT_CLOUD_STITCHER_HPP

#include "types.hpp"
#include "point_cloud_types.hpp"


namespace cuphoto {

enum class PointCloudStitcherBackend {
    FEATURE_ESTIMATOR_LM,
    ICP_ESTIMATOR,
    FEATURE_AND_ICP_ESTIMATOR,
    UNKNOWN
};


struct PointCloudStitcherConfig
{
    /* SIFT config: */
    r32 sift_min_scale = 0.1f;
    i32 sift_n_octaves = 6;
    i32 sift_n_scales_per_octave = 10;
    r32 sift_min_contrast = 0.5f;

    /* descriptor config: */
    r32 desc_normal_radius_search = 0.1;
    r32 desc_feature_radius_search = 0.2;
    i32 desc_inlier_size = 200;
    r32 desc_inlier_threshold = 1.8; // 1.8

    /* ICP config */
    r64 icp_max_correspond_dist = 0.7;
    r64 icp_transformation_eps = 1e-7;
    i32 icp_max_iteration = 50;
    r32 icp_ransac_threshold = 0.06; // 0.05 = default
    r32 icp_resolution_voxel_grid = 0.1;
    r32 icp_step_resolution_point_cloud = 0.07;
    i32 icp_min_points_per_voxel = -1; // not used
};


class PointCloudStitcher
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<PointCloudStitcher>;

    PointCloudStitcher(const PointCloudStitcherBackend pcsb = PointCloudStitcherBackend::FEATURE_AND_ICP_ESTIMATOR);
    PointCloudCPtr stitch(const std::vector<PointCloudCPtr>& pcl_pc,
                          std::vector<Mat4>& transforms,
                          const PointCloudStitcherConfig& pcsc = PointCloudStitcherConfig());
protected:
    void transform_estimation_icp(const std::vector<PointCloudCPtr>& pcl_pc,
                                  std::vector<Mat4f>& transforms,
                                  const PointCloudStitcherConfig& pcsc);
    void transform_estimation_rigid_lm(const std::vector<PointCloudCPtr>& pcl_pc,
                                       std::vector<Mat4f>& transforms,
                                       const PointCloudStitcherConfig& pcsc);
private:
    PointCloudCPtr stitch_by_transformation(const std::vector<PointCloudCPtr>& pcl_pc,
                                            const std::vector<Mat4>& transforms);

private:
    PointCloudStitcherBackend pcsb;
};



};



#endif