#ifndef CUPHOTO_LIB_TYPES_HPP
#define CUPHOTO_LIB_TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Dense>

#include <sophus/se3.hpp>

#include <stdint.h>
#include <chrono>

#include <boost/uuid/uuid.hpp>

namespace cuphoto {

using i8 = int8_t;
using ui8 = uint8_t;
using byte = i8;
using i16 = int16_t;
using ui16 = uint16_t;
using i32 = int32_t;
using i64 = int64_t;
using ui32 = uint32_t;
using ui64 = uint64_t;
using ul = unsigned long;
using r32 = float;
using r64 = double; 
using ch = char;
using uch = unsigned char;

// Eigen matrix and quaternion
using MatXX = Eigen::Matrix<r64, Eigen::Dynamic, Eigen::Dynamic>;
using Mat22 = Eigen::Matrix<r64, 2, 2>;
using Mat33 = Eigen::Matrix<r64, 3, 3>;
using Mat34 = Eigen::Matrix<r64, 3, 4>;
using Mat43 = Eigen::Matrix<r64, 4, 3>;
using Mat44 = Eigen::Matrix<r64, 4, 4>;
using Mat66 = Eigen::Matrix<r64, 6, 6>;
using Mat3 = Mat33;
using Mat4 = Mat44;

using MatXXf = Eigen::Matrix<r32, Eigen::Dynamic, Eigen::Dynamic>;
using Mat22f = Eigen::Matrix<r32, 2, 2>;
using Mat33f = Eigen::Matrix<r32, 3, 3>;
using Mat34f = Eigen::Matrix<r32, 3, 4>;
using Mat43f = Eigen::Matrix<r32, 4, 3>;
using Mat44f = Eigen::Matrix<r32, 4, 4>;
using Mat66f = Eigen::Matrix<r32, 6, 6>;
using Mat3f = Mat33f;
using Mat4f = Mat44f;

using VecX = Eigen::Matrix<r64, Eigen::Dynamic, 1>;
using Vec10 = Eigen::Matrix<r64, 10, 1>;
using Vec9 = Eigen::Matrix<r64, 9, 1>;
using Vec7 = Eigen::Matrix<r64, 7, 1>;
using Vec6 = Eigen::Matrix<r64, 6, 1>;
using Vec5 = Eigen::Matrix<r64, 5, 1>;
using Vec4 = Eigen::Matrix<r64, 4, 1>;
using Vec3 = Eigen::Matrix<r64, 3, 1>;
using Vec2 = Eigen::Matrix<r64, 2, 1>;

using VecXf = Eigen::Matrix<r32, Eigen::Dynamic, 1>;
using Vec10f = Eigen::Matrix<r32, 10, 1>;
using Vec9f = Eigen::Matrix<r32, 9, 1>;
using Vec7f = Eigen::Matrix<r32, 7, 1>;
using Vec6f = Eigen::Matrix<r32, 6, 1>;
using Vec5f = Eigen::Matrix<r32, 5, 1>;
using Vec4f = Eigen::Matrix<r32, 4, 1>;
using Vec3f = Eigen::Matrix<r32, 3, 1>;
using Vec2f = Eigen::Matrix<r32, 2, 1>;

using Vec2i = Eigen::Matrix<i64, 2, 1>;
using Vec3i = Eigen::Matrix<i64, 3, 1>;

using Quat = Eigen::Quaterniond;
using Quatf = Eigen::Quaternionf;


// Sophus
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

// time
using time_point = std::chrono::time_point<std::chrono::system_clock>;
using system_clock = std::chrono::system_clock;

// uuid
using uuid = boost::uuids::uuid;

};


#endif  // CUPHOTO_LIB_TYPES_H