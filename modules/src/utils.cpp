#include "utils.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <ceres/rotation.h>
#include <fstream>


namespace curec {


static void camera_to_center(const r64* camera,
                             r64* center) {
    r64 axis_angle[3];
    Eigen::Map<Eigen::VectorXd> axis_angle_ref(axis_angle, 3);
    ceres::QuaternionToAngleAxis(camera, axis_angle);

    // c = -R't
    Eigen::VectorXd inverse_rotation = -axis_angle_ref;
    ceres::AngleAxisRotatePoint(inverse_rotation.data(),  &(camera[4]), center);
    Eigen::Map<Eigen::VectorXd>(center, 3) *= -1.0;
}

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
    // TODO: check order bgr
    e_color(0) = static_cast<r32>(cv_color[2]);
    e_color(1) = static_cast<r32>(cv_color[1]);
    e_color(2) = static_cast<r32>(cv_color[0]);
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


void write_ply_file(const std::string &filename, const std::vector<SE3>& poses, 
                    const std::vector<Vec3>& pts, const std::vector<Vec3f>& color) {
    std::ofstream of(filename.c_str());
    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << poses.size() + pts.size()
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header\n";
    
    // Export extrinsic data (i.e. camera centers) as red points.
    r64 center[3];
    for (auto i = 0; i < poses.size(); ++i) {
        r64 camera[7];
        Mat3 R = poses[i].rotationMatrix();
        Vec3 t = poses[i].translation();
        Eigen::Quaterniond q(R);
        camera[0] = q.w();
        camera[1] = q.x();
        camera[2] = q.y();
        camera[3] = q.z();
        camera[4] = t.x();
        camera[5] = t.y();
        camera[6] = t.z();
        camera_to_center(camera, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 255 0 0" << '\n';
    }

    for (auto i = 0; i < pts.size(); ++i) {
        of << pts[i].x() << " " << pts[i].y() << " " << pts[i].z() << " ";
        of << color[i].x() << " " << color[i].y() << " " << color[i].z() << "\n";
    }
}

};