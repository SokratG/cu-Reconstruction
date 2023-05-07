#ifndef CUPHOTO_LIB_UTILS_HPP
#define CUPHOTO_LIB_UTILS_HPP

#include "types.hpp"
#include <unistd.h>
#include <utility>
#include <chrono>
#include <string_view>
#include <set>
#include <string>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/functional/hash.hpp>

#include <opencv2/core/core.hpp>

#include <glog/logging.h>

namespace cuphoto {

class TimeProfile
{
public:
    TimeProfile(bool show = false) {
        start = std::chrono::high_resolution_clock::now();
        showTimeDestr = show;
    }
    ~TimeProfile() {
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (showTimeDestr)
            LOG(INFO) << duration.count() << " ms\n"; //µs
    }
    int64_t getTime() noexcept
    {
        end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    bool showTimeDestr = false;
};


class UUID 
{
public:
    static uuid gen();
};


ui64 gen_combined_hash(const ui64 v1, const ui64 v2);

ui64 gen_combined_key(const ui64 v1, const ui64 v2);

Vec3f cv_rgb_2_eigen_rgb(const cv::Vec3b& cv_color);

void write_ply_file(const std::string_view filename, const std::vector<SE3>& poses, 
                    const std::vector<Vec3>& pts, const std::vector<Vec3f>& color);
void write_mesh_to_ply(const std::string& file_name, const std::vector<Vec3f>& vertices, 
                       const std::vector<Vec3i>& triangles);

bool save_image(const std::string& filepath, const cv::InputArray image);

void hwc_to_chw(cv::InputArray src, cv::OutputArray dst);
void chw_to_hwc(cv::InputArray src, cv::OutputArray dst);


std::set<std::string> files_directory(const std::string& data_path, const std::set<std::string>& extensions);

};

namespace std
{
// std::unordered_map<boost::uuids::uuid, T, boost::hash<boost::uuids::uuid>>
template<>
struct hash<boost::uuids::uuid>
{
    std::size_t operator()(boost::uuids::uuid const& uid) const noexcept
    {
        return boost::hash<boost::uuids::uuid>()(uid);
    }
};
}

#endif