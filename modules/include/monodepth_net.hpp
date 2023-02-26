#ifndef CUPHOTO_LIB_MONODEPTH_NET_HPP
#define CUPHOTO_LIB_MONODEPTH_NET_HPP

#include "types.hpp"
#include "CudaUtils/tensorNet.h"

#include "config.hpp"

#include <opencv2/core/cuda.hpp>

#include <string>

namespace cuphoto {

enum class TensorPreprocess {
    NORMALIZATION = 0,
    STANDARDIZATION,
};

class MonoDepthNN : public tensorNet
{
public:
    enum NetworkType {
		MIDAS_V2_SMALL_256 = 0,	/**< Midas v2 backbone */ 
        FCN_MOBILENET, /**< MobileNet backbone */
	};
    struct PreprocessData {
        i32 image_format;
        i32 stride;
        r32 max_intensity;
        float2 norm_range;
        float3 mean;
        float3 stddev;
        bool chw_order;
    };

public:
    MonoDepthNN(const Config& cfg, const NetworkType ntype = NetworkType::FCN_MOBILENET);

    bool process(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const TensorPreprocess trrt_preproc = TensorPreprocess::NORMALIZATION);

    NetworkType network_type_from_str(const std::string& modelname);

    std::string network_type_to_str(const NetworkType modeltype);

private:
    void check_tensorrt_precision_type(const i32 pecision_type);
    void check_image_format(const i32 format);
    void check_tensorrt_device_type(const i32 device_type);
    bool copy_to_tensor(const cv::cuda::GpuMat& input,  const TensorPreprocess trrt_preproc);
private:
    NetworkType network_type;
    i32 input_layer_width;
    i32 input_layer_height;
    i32 output_layer_width;
    i32 output_layer_height;
    i32 batch_size;
    PreprocessData preproc;
};


}


#endif