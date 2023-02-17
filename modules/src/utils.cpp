#include "utils.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <ceres/rotation.h>
#include <fstream>

#include <filesystem>


namespace cuphoto {

static boost::uuids::random_generator uuid_generator;

uuid UUID::gen() {
    return uuid_generator();
}

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
    // TODO: check order rgb/bgr
    e_color(0) = static_cast<r32>(cv_color[0]);
    e_color(1) = static_cast<r32>(cv_color[1]);
    e_color(2) = static_cast<r32>(cv_color[2]);
    return e_color;
}


void write_ply_file(const std::string_view filename, const std::vector<SE3>& poses, 
                    const std::vector<Vec3>& pts, const std::vector<Vec3f>& color) {
    std::ofstream of(std::string(filename).c_str());
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


std::set<std::string> files_directory(const std::string& data_path, const std::set<std::string>& extensions) {
    std::set<std::string> files;
    const std::filesystem::path dir{data_path};
    for (const auto& entry : std::filesystem::directory_iterator(dir, 
         std::filesystem::directory_options::skip_permission_denied)) {

        const std::string ext = entry.path().extension();
        auto find_res = extensions.find(ext);
        if (find_res == extensions.end())
            continue;
        
        const std::filesystem::file_status ft(status(entry));
        const auto type = ft.type();
        if (type == std::filesystem::file_type::directory ||
            type == std::filesystem::file_type::fifo || 
            type == std::filesystem::file_type::socket ||
            type == std::filesystem::file_type::unknown) {
			continue;
        } else {
            files.insert(canonical(entry.path()).string());
        }
    }
    return files;
}

};