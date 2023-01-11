#include "utils.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace curec {

ui64 gen_combined_hash(const ui64 v1, const ui64 v2) 
{
    ui64 seed = 0;
    boost::hash_combine(seed, v1);
    boost::hash_combine(seed, v2);
    return seed;
}

ui64 gen_combined_key(const ui64 v1, const ui64 v2) 
{
     return ((v1 << 32) + v2);
}


Vec3f cv_rgb_2_eigen_rgb(const cv::Vec3b& cv_color) 
{
    Vec3f e_color;
    // TODO: check order rgb/bgr
    e_color(0) = static_cast<r32>(cv_color[0]);
    e_color(1) = static_cast<r32>(cv_color[1]);
    e_color(2) = static_cast<r32>(cv_color[2]);
    return e_color;
}

bool triangulation(const std::vector<SE3> &poses, 
                   const std::vector<Vec3> points,
                   const r64 confidence_thrshold,
                   Vec3 &pt_world) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (auto i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1); 
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < confidence_thrshold) {
        return true;
    }
    return false;
}

};