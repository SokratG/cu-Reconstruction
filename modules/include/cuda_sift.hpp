#ifndef CUREC_LIB_CUDA_SIFT_WRAPPER_HPP
#define CUREC_LIB_CUDA_SIFT_WRAPPER_HPP

#include "types.hpp"
#include "CudaSift/cudaSift.h"
#include <opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d.hpp>

namespace curec {

struct SiftParams {
    i32 maxKeypoints = 1000;
    i32 numOctaves = 5;
	r32 initBlur = 1.0f;
	r32 thresh = 2.0f; //3.5f;
	r32 minScale = 0.0f;
	bool upScale = false;
};

// source: https://github.com/Celebrandil/CudaSift/issues/42
class CudaSiftWrapper : public cv::Feature2D {
public:
    CudaSiftWrapper(const SiftParams& sp = SiftParams());
	~CudaSiftWrapper();

    static cv::Ptr<CudaSiftWrapper> create(const SiftParams& sp = SiftParams());

    void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,  
                          cv::OutputArray descriptors, bool useProvidedKeypoints = false) override;

    int descriptorSize() const override;
    int descriptorType() const override;
    int defaultNorm() const override;
    cv::String getDefaultName() const override;
    bool empty() const override;

    void read(const cv::FileNode& fn) override;
    void write(cv::FileStorage& fs) const override;

private:
    SiftData siftData;
    SiftParams sp;
};

using CudaSift = CudaSiftWrapper;

};

#endif