#ifndef CUREC_LIB_TYPES_HPP
#define CUREC_LIB_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/se3.hpp>

#include <stdint.h>
#include <chrono>

#include <boost/uuid/uuid.hpp>

using i32 = int32_t;
using i64 = int64_t;
using ui32 = uint32_t;
using ui64 = uint64_t;
using ul = unsigned long;
using r32 = float;
using r64 = double; 
using ch = char;
using uch = unsigned char;

// Eigen matrix
using MatXX = Eigen::Matrix<r64, Eigen::Dynamic, Eigen::Dynamic>;
using Mat22 = Eigen::Matrix<r64, 2, 2>;
using Mat33 = Eigen::Matrix<r64, 3, 3>;
using Mat34 = Eigen::Matrix<r64, 3, 4>;
using Mat43 = Eigen::Matrix<r64, 4, 3>;
using Mat44 = Eigen::Matrix<r64, 4, 4>;
using Mat66 = Eigen::Matrix<r64, 6, 6>;

using MatXXf = Eigen::Matrix<r32, Eigen::Dynamic, Eigen::Dynamic>;
using Mat22f = Eigen::Matrix<r32, 2, 2>;
using Mat33f = Eigen::Matrix<r32, 3, 3>;
using Mat34f = Eigen::Matrix<r32, 3, 4>;
using Mat43f = Eigen::Matrix<r32, 4, 3>;
using Mat44f = Eigen::Matrix<r32, 4, 4>;
using Mat66f = Eigen::Matrix<r32, 6, 6>;

using VecX = Eigen::Matrix<r64, Eigen::Dynamic, 1>;
using Vec9 = Eigen::Matrix<r64, 9, 1>;
using Vec6 = Eigen::Matrix<r64, 6, 1>;
using Vec5 = Eigen::Matrix<r64, 5, 1>;
using Vec4 = Eigen::Matrix<r64, 4, 1>;
using Vec3 = Eigen::Matrix<r64, 3, 1>;
using Vec2 = Eigen::Matrix<r64, 2, 1>;

using VecXf = Eigen::Matrix<r32, Eigen::Dynamic, 1>;
using Vec9f = Eigen::Matrix<r32, 9, 1>;
using Vec6f = Eigen::Matrix<r32, 6, 1>;
using Vec5f = Eigen::Matrix<r32, 5, 1>;
using Vec4f = Eigen::Matrix<r32, 4, 1>;
using Vec3f = Eigen::Matrix<r32, 3, 1>;
using Vec2f = Eigen::Matrix<r32, 2, 1>;

// Sophus
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

// time
using time_point = std::chrono::time_point<std::chrono::system_clock>;
using system_clock = std::chrono::system_clock;

// uuid
using uuid = boost::uuids::uuid;

#endif  // CUREC_LIB_TYPES_H