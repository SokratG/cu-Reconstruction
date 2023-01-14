#ifndef CUREC_LIB_UTILS_HPP
#define CUREC_LIB_UTILS_HPP

#include "types.hpp"
#include <unistd.h>
#include <utility>
#include <chrono>
#include <string_view>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/functional/hash.hpp>

#include <opencv2/core/core.hpp>

#include <glog/logging.h>

namespace curec {

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
    static boost::uuids::random_generator gen;
};


ui64 gen_combined_hash(const ui64 v1, const ui64 v2);

ui64 gen_combined_key(const ui64 v1, const ui64 v2);

Vec3f cv_rgb_2_eigen_rgb(const cv::Vec3b& cv_color);

bool triangulation(const SE3& src_pose,
                   const SE3& dst_pose,
                   const std::pair<Vec3, Vec3>& points,
                   const r64 confidence_thrshold,
                   Vec3 &pt_world);

void write_ply_file(const std::string_view filename, const std::vector<SE3>& poses, 
                    const std::vector<Vec3>& pts, const std::vector<Vec3f>& color);

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