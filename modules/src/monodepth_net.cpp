#include "monodepth_net.hpp"
#include "cuda/tensorrt_utils.cuh"

#include "cp_exception.hpp"

#include <opencv2/cudawarping.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/algorithm/string.hpp>
#include <glog/logging.h>

namespace cuphoto {

constexpr const char* DEPTH_BLOB_DEFAULT_INPUT = "inputs:0"; // "input_0" - midas
constexpr const char* DEPTH_BLOB_DEFAULT_OUTPUT = "Identity:0"; // "output_0" - midas
constexpr i32 DEFAULT_BATCH_SIZE = 1;
constexpr bool allow_GPU_fallback = true;


MonoDepthNN::MonoDepthNN(const Config& cfg, const NetworkType ntype) : network_type(ntype) {
	input_layer_width = cfg.get<i32>("modelnetwork.layer.input.width");
	input_layer_height = cfg.get<i32>("modelnetwork.layer.input.height");
	output_layer_width = cfg.get<i32>("modelnetwork.layer.output.width");
	output_layer_height = cfg.get<i32>("modelnetwork.layer.output.height");
	batch_size = cfg.get<i32>("modelnetwork.layer.input.batch_size", DEFAULT_BATCH_SIZE);

	preproc.image_format = cfg.get<i32>("modelnetwork.layer.input.image_format", 0);
	check_image_format(preproc.image_format);

	preproc.norm_range = make_float2(cfg.get<r32>("modelnetwork.layer.input.normalization.min", 0.0f),
							 		 cfg.get<r32>("modelnetwork.layer.input.normalization.max", 1.0f));

	preproc.mean = make_float3(cfg.get<r32>("modelnetwork.layer.input.standardization.mean.x", 0.0f),
					   		   cfg.get<r32>("modelnetwork.layer.input.standardization.mean.y", 0.0f),
					   		   cfg.get<r32>("modelnetwork.layer.input.standardization.mean.z", 0.0f));
	
	preproc.stddev = make_float3(cfg.get<r32>("modelnetwork.layer.input.standardization.stddev.x", 1.0f),
					   	 		 cfg.get<r32>("modelnetwork.layer.input.standardization.stddev.y", 1.0f),
					   	 		 cfg.get<r32>("modelnetwork.layer.input.standardization.stddev.z", 1.0f));
	preproc.max_intensity = cfg.get<r32>("modelnetwork.layer.input.max_intensity", MAX_INTENSITY);
	preproc.stride = cfg.get<i32>("modelnetwork.layer.input.stride", 0);
	preproc.chw_order = static_cast<bool>(cfg.get<i32>("modelnetwork.layer.input.order", 0));
	

	i32 prec_type = cfg.get<i32>("tensorrt.precision.type", 1);
	check_tensorrt_precision_type(prec_type);
	i32 dev_type = cfg.get<i32>("tensorrt.device.type", 0);
	check_tensorrt_device_type(dev_type);
	precisionType precision_type = static_cast<precisionType>(prec_type);
	deviceType device_type = static_cast<deviceType>(dev_type);

	std::string input_blob_name = cfg.get<std::string>("modelnetwork.layer.input.name", DEPTH_BLOB_DEFAULT_INPUT);
	std::string output_blob_name = cfg.get<std::string>("modelnetwork.layer.output.name", DEPTH_BLOB_DEFAULT_OUTPUT);
	std::string modelpath = cfg.get<std::string>("modelnetwork.path");

    if (!this->LoadNetwork(nullptr, modelpath.c_str(), nullptr, 
                           input_blob_name.c_str(), output_blob_name.c_str(),
                           batch_size, precision_type, device_type, allow_GPU_fallback)) {
		throw CuPhotoException("Can't load given network model into TensorRT: " + modelpath);
    }
}


MonoDepthNN::NetworkType MonoDepthNN::network_type_from_str(const std::string& modelname)
{
	if(modelname.empty())
		throw CuPhotoException("The model name is empty!");

	MonoDepthNN::NetworkType type = MonoDepthNN::NetworkType::FCN_MOBILENET;

	if(boost::iequals(modelname, "monodepth_fcn_mobilenet"))
		type = MonoDepthNN::NetworkType::FCN_MOBILENET;
    else if(boost::iequals(modelname, "midas_v2_small_256"))
		type = MonoDepthNN::NetworkType::MIDAS_V2_SMALL_256;
	else
		throw CuPhotoException("The given model name does not implemented!");

	return type;
}


std::string MonoDepthNN::network_type_to_str(const MonoDepthNN::NetworkType modeltype) {
    switch(modeltype) {
		case FCN_MOBILENET:	return "MonoDepth_FCN_Mobilenet";
        case MIDAS_V2_SMALL_256:	return "midas_v2_small_256";
		default:			throw CuPhotoException("The given type has not implemented!");
	}
}

void MonoDepthNN::check_tensorrt_device_type(const i32 device_type) {
	if (device_type > 2 || device_type < 0) {
		throw CuPhotoException("The given tensorrt device type is not valid!");
	}
}

void MonoDepthNN::check_tensorrt_precision_type(const i32 pecision_type) {
	if (pecision_type > 4 || pecision_type < 0) {
		throw CuPhotoException("The given tensorrt precision type is not valid!");
	}
}

void MonoDepthNN::check_image_format(const i32 format) {
	if (format > 3 || format < 0) {
		throw CuPhotoException("The given image format is not valid!");
	}
}

bool MonoDepthNN::copy_to_tensor(const cv::cuda::GpuMat& input,  const TensorPreprocess trrt_preproc) {
	switch (trrt_preproc) {
		case TensorPreprocess::NORMALIZATION:
			if(CUDA_FAILED(copy_tensor_normalization_RGB(input, mInputs[0].CUDA, input_layer_width, input_layer_height, 
									   		 		 	 preproc.norm_range, GetStream(), 
														 preproc.max_intensity, preproc.stride, preproc.chw_order))) {
				return false;
			}
			break;
		case TensorPreprocess::STANDARDIZATION:
			if(CUDA_FAILED(copy_tensor_standardization_RGB(input, mInputs[0].CUDA, input_layer_width, input_layer_height, 
									   		 		 	   preproc.mean, preproc.stddev, GetStream(), 
														   preproc.stride, preproc.chw_order))) {
				return false;
			}
			break;
		default:
			throw CuPhotoException("The given type of preprocess flag is not valid!");
	}
	return true;
}


bool MonoDepthNN::process(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, const TensorPreprocess trrt_preproc) {
	cv::cuda::GpuMat resized_input;
	cv::cuda::resize(input, resized_input, cv::Size(input_layer_width, input_layer_height), cv::INTER_CUBIC);

	if(IsModelType(MODEL_ONNX)) {
		// remap from [0.0, 255.0] -> mean pixel subtraction or std dev applied
		const bool copy_result = copy_to_tensor(resized_input, trrt_preproc);
		if (!copy_result) {
			LOG(ERROR) << "Can't copy data into tensor with RGB data";
			return false;
		}
	} else {
		LOG(ERROR) << "The given model format is not supported! Supported format: ONNX";
		return false;
	}

	if(!ProcessNetwork(true))
		return false;

	output = cv::cuda::GpuMat(cv::Size(output_layer_width, output_layer_height), CV_32F, mOutputs[0].CUDA);

    return true;
}

}