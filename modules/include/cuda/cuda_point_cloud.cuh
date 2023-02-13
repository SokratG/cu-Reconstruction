#ifndef CUPHOTO_LIB_POINT_CLOUD_CUH
#define CUPHOTO_LIB_POINT_CLOUD_CUH

#include "types.cuh"

#include <opencv2/core/cuda/common.hpp>

#include <memory>
#include <array>

namespace cuphoto {

class cudaPointCloud
{
public:
	struct Vertex
	{
		using Ptr = std::shared_ptr<Vertex>;
		// The XYZ position of the point.
		float3 pos;

		/**
		 * The RGB color of the point.
		 * @note will be white if RGB data not provided
		 */
		uchar3 color;

		/**
		 * The class ID of the point.
		 * @note will be 0 if classification data no provided
		 */
         ui16 classID;

	} __attribute__((packed));

public:
	using Ptr = std::shared_ptr<cudaPointCloud>;

	~cudaPointCloud();

	static cudaPointCloud::Ptr create(const std::array<r64, 9>& camera_mat, const size_t total_number_pts);

	inline ui64 get_total_num_points() const { return total_num_pts; }

	inline ui64 get_num_points() const { return num_pts; }

	inline ui64 get_size() const { return num_pts * sizeof(Vertex); }

	inline Vertex* get_data(const ui64 index) const	{ return device_pts + index; }

	inline Vertex* get_points() const { return device_pts; }

	inline void set_current_pts() {num_pts = total_num_pts;}

	bool extract_points(const cv::cuda::PtrStepSzf depth,
						const cv::cuda::PtrStepb colors,
						const std::array<r64, 7>& transform,
						const i32 frame_idx);

	void filter_depth(const r32 depth_threshold);

	void clear();
	void free();
	bool save_ply(const std::string& filepath) const;
protected:
	bool reserve_memory(const ui64 query_number_points);

	cudaPointCloud(const std::array<r64, 9>& camera_mat, const size_t total_number_pts);

private:
	Vertex* device_pts;
	ui64 num_pts;
	ui64 total_num_pts;
	
	bool hasRGB;
};

};

#endif