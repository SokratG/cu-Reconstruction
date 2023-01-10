#ifndef CUREC_LIB_SFM_HPP
#define CUREC_LIB_SFM_HPP

#include "keyframe.hpp"
#include "camera.hpp"
#include <opencv2/core/core.hpp>
#include <vector>

namespace curec {


class Sfm {
public:
    Sfm(const Camera::Ptr camera);
    bool add_frame(const cv::Mat frame);
    void build_landmark_graph(); 
private:
    std::vector<KeyFrame::Ptr> frames;
    Camera::Ptr camera;
};


};

#endif // 