#ifndef CUREC_LIB_ME_HPP
#define CUREC_LIB_ME_HPP

#include "types.hpp"
#include <memory>

namespace curec {

class MotionEstimation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimation>;

    MotionEstimation() {}

    bool estimate_motion_ransac();
    bool estimate_motion_non_lin_opt();
private:
    // TODO
};

};


#endif