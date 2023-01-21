#include "cuda_sift.hpp"
#include "CudaSift/cudaImage.h"
#include "cp_exception.hpp"
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace cuphoto {

constexpr auto SIFT_DESC_DIM = 128;

CudaSiftWrapper::CudaSiftWrapper(const SiftParams& _sp) : sp(_sp) {
	InitSiftData(siftData, sp.maxKeypoints, true, true);
}

cv::Ptr<CudaSiftWrapper> CudaSiftWrapper::create(const SiftParams& sp) {
    return cv::makePtr<CudaSiftWrapper>(sp);
}

CudaSiftWrapper::~CudaSiftWrapper() {
    FreeSiftData(siftData);
}

int CudaSiftWrapper::descriptorSize() const {
    return SIFT_DESC_DIM;
}

int CudaSiftWrapper::descriptorType() const {
    return CV_32F;
}

int CudaSiftWrapper::defaultNorm() const {
    return cv::NORM_L2;
}

cv::String CudaSiftWrapper::getDefaultName() const {
    return (Feature2D::getDefaultName() + ".SIFT");
}

bool CudaSiftWrapper::empty() const {
    throw CuPhotoException("Method not implemented!");
}

void CudaSiftWrapper::read(const cv::FileNode& fn) {
    throw CuPhotoException("Method not implemented!");
}

void CudaSiftWrapper::write(cv::FileStorage& fs) const {
    throw CuPhotoException("Method not implemented!");
}


void CudaSiftWrapper::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,  
                                       cv::OutputArray descriptors, bool useProvidedKeypoints)
{
    /*
        cv::InputArray mask: not used
        bool useProvidedKeypoints: not used
    */ 
   	cv::cuda::GpuMat refCudaImage = image.getGpuMat();
	cv::cuda::GpuMat& refCudaDesc = descriptors.getGpuMatRef();
    if(refCudaImage.empty() || refCudaImage.depth() != CV_8U)
        throw CuPhotoException("image is empty or has incorrect depth (!=CV_8U)");

	cv::Mat tmpImage, cpu_descriptor;
	refCudaImage.download(tmpImage);
	tmpImage.convertTo(tmpImage, CV_32FC1);

	CudaImage cudaImg;
	// TODO: add cuda::GpuMat.data;
	cudaImg.Allocate(refCudaImage.cols, refCudaImage.rows, iAlignUp(image.size().width, SIFT_DESC_DIM), false, NULL, (r32*)tmpImage.data);
	cudaImg.Download();

	ExtractSift(siftData, cudaImg, sp.numOctaves, sp.initBlur, sp.thresh, sp.minScale, sp.upScale);

	// Convert SiftData to Keypoints
	keypoints.resize(siftData.numPts);
	cv::parallel_for_(cv::Range(0, siftData.numPts), [&](const cv::Range& range) {
		for (i32 r = range.start; r < range.end; r++) {
			keypoints[r] = cv::KeyPoint(cv::Point2f(siftData.h_data[r].xpos, siftData.h_data[r].ypos), siftData.h_data[r].scale, siftData.h_data[r].orientation, siftData.h_data[r].score, siftData.h_data[r].subsampling, siftData.h_data[r].match);
		}
	});

	// Convert SiftData to Mat Descriptor
	std::vector<r32> data;
	for (i32 i = 0; i < siftData.numPts; i++) {
		data.insert(data.end(), siftData.h_data[i].data, siftData.h_data[i].data + SIFT_DESC_DIM);
	}

	cv::Mat tempDescriptor(siftData.numPts, SIFT_DESC_DIM, CV_32FC1, &data[0]);
	refCudaDesc.upload(tempDescriptor);
}

};