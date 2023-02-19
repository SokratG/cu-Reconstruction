#include "cuda/cuda_point_cloud.cuh"
#include "cuda/se3.cuh"

#include "CudaUtils/cudaUtility.cuh"
#include "CudaUtils/logging.h"

#include <numeric>
#include <fstream>

namespace cuphoto {

__device__ r64 K[3][3] {0};


template<bool useRGB = false>
__global__ void point_cloud_extract(const cv::cuda::PtrStepSzf depth, 
                                    const cv::cuda::PtrStepb color,
                                    const float4 quat,  const float3 trans,
                                    const i64 start_idx,
                                    cudaPointCloud::Vertex* points) 
{
    const ui32 width = depth.cols;
    const ui32 height = depth.rows;
    const i32 x = blockIdx.x * blockDim.x + threadIdx.x;
	const i32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
		return;
    
    const r32 depth_value = depth.ptr(y)[x];

    cudaPointCloud::Vertex point;
    point.classID = 0;

    SE3<r32> transform(quat, trans);

    const r64 fx = K[0][0];
    const r64 fy = K[1][1];
    const r64 cx = K[2][0];
    const r64 cy = K[2][1];

    point.pos = make_float3((r32(x) - cx) * depth_value / fx, 
                            (r32(y) - cy) * depth_value / fy,
                            depth_value);

    if (depth_value <= 0.0f) {
        point.color = make_uchar3(0, 0, 0);
        return;
    }

    point.pos = transform * point.pos;

    // for opengl / meshlab
    point.pos.y *= -1.f;
    point.pos.z *= -1.f;

    if (useRGB) {
        const uchar3 rgb = ((uchar3*)color.ptr(y))[x];
        point.color = make_uchar3(rgb.x, rgb.y, rgb.z);
    } else
		point.color = make_uchar3(255, 255, 255);

    const i64 pidx = start_idx + y * width + x;
    points[pidx] = point;
}


cudaPointCloud::cudaPointCloud(const std::array<r64, 9>& camera_param, const size_t total_number_pts) 
                                : device_pts(nullptr), hasRGB(false), num_pts(0)
{
    camera_matrix = camera_param;
    const bool alloc_mem_res = reserve_memory(total_number_pts);

    if (!alloc_mem_res) {
        LogError(LOG_CUDA "cudaPointCloud::reserve_memory() -- can't allocate query CUDA memory %lu\n", total_number_pts);
        throw std::bad_alloc();
    }

    total_num_pts = total_number_pts;
}

cudaPointCloud::~cudaPointCloud() {
    free();
}

cudaPointCloud::Ptr cudaPointCloud::create(const std::array<r64, 9>& camera_mat,
                                           const size_t total_number_pts) {
    return std::shared_ptr<cudaPointCloud>(new cudaPointCloud(camera_mat, total_number_pts));
}


cudaPointCloud::Ptr cudaPointCloud::merge(const cudaPointCloud::Ptr pc1, const cudaPointCloud::Ptr pc2) {
    const ui64 query_num_pts = pc1->get_num_points() + pc2->get_num_points();
    cudaPointCloud::Ptr cuda_pc = cudaPointCloud::create(pc1->get_camera_parameters(), query_num_pts);
    Vertex* cuda_pts = cuda_pc->get_points();

    CUDA(cudaMemcpy(cuda_pts, pc1->get_points(), pc1->get_size(), cudaMemcpyDefault));

    CUDA(cudaMemcpy(cuda_pts + pc1->get_num_points(), pc2->get_points(), pc2->get_size(), cudaMemcpyDefault));

    cuda_pc->set_total_number_pts();

    return cuda_pc;
}


void cudaPointCloud::clear() {
    if (device_pts != nullptr) {
        CUDA(cudaMemset(device_pts, 0, get_size()));
    }
    num_pts = 0;
}

void cudaPointCloud::free() {
    if (device_pts != nullptr) {
        CUDA(cudaFree(device_pts));
    }
    device_pts = nullptr;
    num_pts = 0;
}

bool cudaPointCloud::reserve_memory(const ui64 query_number_points) {
    if (query_number_points <= 0)
        return true;

    if (query_number_points <= num_pts && device_pts != nullptr) {
        clear();
        return true;
    }

    free();

    const size_t query_size = query_number_points * sizeof(Vertex);

    if (CUDA_FAILED(cudaMallocManaged(&device_pts, query_size, cudaMemAttachGlobal))) {
        return false;
    }

    cudaDeviceSynchronize();

	return true;
}

bool cudaPointCloud::extract_points(const cv::cuda::PtrStepSzf depth, 
                                    const cv::cuda::PtrStepb color,
                                    const std::array<r64, 7>& T,
                                    const i32 frame_idx)
{
    if (!depth) {
        LogError(LOG_CUDA "cudaPointCloud::extract_points() -- depth map is null pointer\n");
		return false;
    }

    const ui32 width = depth.cols;
    const ui32 height = depth.rows;


    if (width == 0 || height == 0) {
        LogError(LOG_CUDA "cudaPointCloud::extract_points() -- depth width/height parameters are zero\n");
        return false;
    }

    const dim3 blockDim(8, 8, 1);
	const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);

    if (color) {
        hasRGB = true;
    }

    const i64 start_idx = frame_idx * width * height;

    float4 quat = make_float4(T[1], T[2], T[3], T[0]);
    float3 trans = make_float3(T[4], T[5], T[6]);
    CUDA(cudaMemcpyToSymbol(K, camera_matrix.data(), camera_matrix.size() * sizeof(r64)));

    if (hasRGB)
        point_cloud_extract<true><<<gridDim, blockDim>>>(depth, color, quat, trans, start_idx, device_pts);
    else
        point_cloud_extract<<<gridDim, blockDim>>>(depth, color, quat, trans, start_idx, device_pts);

    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "cudaPointCloud::extract_points() -- failed to extract point cloud with CUDA\n");
        return false;
    }

    num_pts += width * height;

    CUDA(cudaDeviceSynchronize());

    return true;
}

bool cudaPointCloud::add_vertex(const Vertex& v, const ui64 idx) {
    if (idx >= total_num_pts) {
        LogError(LOG_CUDA "cudaPointCloud::add_vertex() -- index out of range\n");
        return false;
    }
    device_pts[idx] = v;
    return true;
}

bool cudaPointCloud::add_vertex(const float3 pos, const uchar3 color, const ui64 idx) {
    if (idx >= total_num_pts) {
        LogError(LOG_CUDA "cudaPointCloud::add_vertex() -- index out of range\n");
        return false;
    }
    device_pts[idx].pos = pos;
    device_pts[idx].color = color;
    return true;
}

void cudaPointCloud::filter_depth(const r32 depth_threshold_min, const r32 depth_threshold_max) {

    const ui64 filter_num_pts = std::accumulate(device_pts, device_pts + num_pts, (ui64)0, 
        [&](const i64 filter_num, const Vertex& v) {
            if (v.pos.z <= depth_threshold_min || v.pos.z >= depth_threshold_max)
                return filter_num + 1;
            return filter_num;
    });
    
    Vertex* filtered_pts;
    const size_t query_size = filter_num_pts * sizeof(Vertex);
    if (CUDA_FAILED(cudaMallocManaged(&filtered_pts, query_size, cudaMemAttachGlobal))) {
        LogError(LOG_CUDA "cudaPointCloud::filter_depth() -- can't allocate query CUDA memory %lu\n", filter_num_pts);
        throw std::bad_alloc();
    }

    for (i64 s_idx = 0, t_idx = 0; s_idx < num_pts; ++s_idx) {
        if (device_pts[s_idx].pos.z > depth_threshold_min && device_pts[s_idx].pos.z < depth_threshold_max)
            continue;
        filtered_pts[t_idx] = device_pts[s_idx];
        t_idx += 1;
    }
    free();

    device_pts = filtered_pts;
    num_pts = filter_num_pts;
}


bool cudaPointCloud::add_point_cloud(const cudaPointCloud::Ptr cuda_pc) {
    const i64 query_num_pts = cuda_pc->get_num_points() + num_pts;
    const ui64 query_mem_size = query_num_pts * sizeof(Vertex);
    Vertex* cuda_pts = nullptr;
    if (CUDA_FAILED(cudaMallocManaged(&cuda_pts, query_mem_size, cudaMemAttachGlobal)))
        return false;

    CUDA(cudaMemcpy(cuda_pts, device_pts, get_size(), cudaMemcpyDefault));

    CUDA(cudaMemcpy(cuda_pts + num_pts, cuda_pc->get_points(), cuda_pc->get_size(), cudaMemcpyDefault));

    free();

    device_pts = cuda_pts;
    num_pts = query_num_pts;
    total_num_pts = query_num_pts;

    return true;
}


bool cudaPointCloud::save_ply(const std::string& filepath) const {
    if (filepath.empty()) {
        LogError(LOG_CUDA "cudaPointCloud::save_ply() -- empty file path for store point cloud\n");
        return false;
    }

    if (!device_pts || num_pts <= 0) {
        LogError(LOG_CUDA "cudaPointCloud::save_ply() -- can't store, point cloud is empty\n");
		return false;
    }

    std::ofstream ply_of(filepath.c_str());
    ply_of << "ply"
           << '\n' << "format ascii 1.0"
           << '\n' << "element vertex " << num_pts
           << '\n' << "property float x"
           << '\n' << "property float y"
           << '\n' << "property float z"
           << '\n' << "property uchar red"
           << '\n' << "property uchar green"
           << '\n' << "property uchar blue"
           << '\n' << "end_header\n";

    for (i64 idx = 0; idx < num_pts; ++idx) {
        r32 x = device_pts[idx].pos.x, y = device_pts[idx].pos.y, z = device_pts[idx].pos.z;
        i32 r = (i32)device_pts[idx].color.x, g = (i32)device_pts[idx].color.y, b = (i32)device_pts[idx].color.z;
        ply_of << x << " " << y << " " << z << " ";
        ply_of << r << " " << g << " " << b << "\n";
    }

    ply_of.close();

    return true;
}

};