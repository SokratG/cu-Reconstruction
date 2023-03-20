#ifndef CUPHOTO_LIB_POINT_CLOUD_CUH
#define CUPHOTO_LIB_POINT_CLOUD_CUH

#include "types.cuh"

#include <opencv2/core/cuda/common.hpp>

#include <memory>
#include <array>
#include <limits>

namespace cuphoto {

class cudaPointCloud
{
public:
	struct Vertex
	{
		using Ptr = Vertex*;
		// The XYZ position of the point.
		float3 pos;
		float4 normal;

		uchar3 color;

        ui16 classID = 0;

	} __attribute__((packed));

public:
	using Ptr = std::shared_ptr<cudaPointCloud>;

	~cudaPointCloud();

	static cudaPointCloud::Ptr create(const std::array<r64, 9>& camera_mat, const size_t total_number_pts);

	static cudaPointCloud::Ptr merge(const cudaPointCloud::Ptr pc1, const cudaPointCloud::Ptr pc2);

	inline ui64 get_total_num_points() const { return total_num_pts; }

	inline ui64 get_num_points() const { return num_pts; }

	inline ui64 get_size() const { return num_pts * sizeof(Vertex); }

	inline Vertex* get_data(const ui64 index) const	{ return device_pts + index; }

	inline Vertex* get_points() const { return device_pts; }

	inline std::array<r64, 9> get_camera_parameters() { return camera_matrix; }

	inline void set_total_number_pts() { num_pts = total_num_pts; }

	bool add_point_cloud(const cudaPointCloud::Ptr cuda_pc);

	bool extract_points(const cv::cuda::PtrStepSzf depth,
						const cv::cuda::PtrStepb colors,
						const std::array<r64, 7>& transform,
						const i32 frame_idx);

	bool transform(const std::array<r64, 7>& transform);

	void filter_depth(const r32 depth_threshold_min = std::numeric_limits<r32>::min(), 
					  const r32 depth_threshold_max = std::numeric_limits<r32>::max());
	bool add_vertex(const Vertex& v, const ui64 idx);
	bool add_vertex(const float3 pos, const uchar3 color, const ui64 idx, const float4 normal = make_float4(0., 0., 0., 0.));
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
	std::array<r64, 9> camera_matrix;
	
	bool hasRGB;
};

};

#endif