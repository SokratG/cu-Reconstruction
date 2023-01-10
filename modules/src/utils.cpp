#include "utils.hpp"

namespace curec {

ui64 gen_combined_key(const ui64 v1, const ui64 v2) 
{
    ui64 seed = 0;
    boost::hash_combine(seed, v1);
    boost::hash_combine(seed, v2);
    return seed;
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

};