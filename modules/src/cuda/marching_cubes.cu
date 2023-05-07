#include "marching_cubes.cuh"
#include "marching_cubes_tables.cuh"

#include "CudaUtils/cudaUtility.cuh"
#include "CudaUtils/cudaMath.cuh"

#include "math_constants.h"

namespace cuphoto {
// TODO: use optimized version - https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/marchingCubes/marchingCubes.cpp

constexpr ui32 NUM_CUBE_SIDE_VERT = 12;
constexpr ui32 NUM_CUBE_TRIANGLES = 5;


__device__ void mc_swap(float3& a, float3& b) {
    float3 temp = a;
    a = b;
    b = temp;
}


__device__ float3 compute_intersection(const i32 edge_idx, const r32 voxel_values[8],
                                       const float3 cube_vertices[8]) {
	i32 v1_index = MC_EDGE_VERTICES[edge_idx][0];
	i32 v2_index = MC_EDGE_VERTICES[edge_idx][1];

	float3 start_vertex = cube_vertices[v1_index];
	r32 start_weight = voxel_values[v1_index];

	float3 end_vertex = cube_vertices[v2_index];
	r32 end_weight = voxel_values[v2_index];

	if ((start_weight > 0) &&  (end_weight < 0)) {
		// Swap start and end
		mc_swap(start_vertex, end_vertex);

		const r32 temp = start_weight;
		start_weight = end_weight;
		end_weight = temp;
	}

	const r32 ratio = (-start_weight) / (end_weight - start_weight);

	// Work out where this lies
	const float3 edge = end_vertex - start_vertex;
	const float3 offset = ratio * edge;
	const float3 intersection = start_vertex + offset;

	return intersection;
}

__device__ i32 detect_edge_intersects(const ui8 voxel_type, const r32 voxel_values[8],
                                     const float3 cube_verts[8], float3 intersection_verts[12]) {

    i32 num_edges_impacted = 0;
    if ((voxel_type != 0) && (voxel_type != 0xFF)) {
        ui16 edge_flags = MC_EDGE_TABLE[voxel_type];
        ui16 mask = 0x01;
        for (i32 idx = 0; idx < NUM_CUBE_SIDE_VERT; idx++) {
			if ((edge_flags & mask ) > 0) {
				intersection_verts[idx] = compute_intersection(idx, voxel_values, cube_verts);

				num_edges_impacted++;
			} else {
				intersection_verts[idx].x = CUDART_NAN_F;
				intersection_verts[idx].y = CUDART_NAN_F;
				intersection_verts[idx].z = CUDART_NAN_F;
			}
			mask = mask << 1;
		}
    }
    return num_edges_impacted;
}

__host__ __device__ ui8 define_voxel_type(const r32 voxel_values[8]) {
	ui8 mask = 0x01, voxel_type = 0x00;
	for (i32 idx = 0; idx < 8; ++idx) {
		if (voxel_values[idx] < 0.f)
			voxel_type = voxel_type | mask;
		mask = mask << 1;
	}
	return voxel_type;
}

__device__ void compute_voxel_vertices(const i32 vx, const i32 vy, const ui32 grid_width, 
                                       const TSDFVolume::TransformationVoxel::Ptr layer_vertices1,
                                       const TSDFVolume::TransformationVoxel::Ptr layer_vertices2, 
                                       float3 voxel_vertices[8]) {
	const i32 base_index = vy * grid_width + vx;
	voxel_vertices[0] = layer_vertices1[base_index].translation;
	voxel_vertices[1] = layer_vertices1[base_index + 1].translation;
	voxel_vertices[2] = layer_vertices2[base_index + 1].translation;
	voxel_vertices[3] = layer_vertices2[base_index].translation;
	voxel_vertices[4] = layer_vertices1[base_index + grid_width].translation;
	voxel_vertices[5] = layer_vertices1[base_index + grid_width + 1].translation;
	voxel_vertices[6] = layer_vertices2[base_index + grid_width + 1].translation;
	voxel_vertices[7] = layer_vertices2[base_index + grid_width].translation;
}


__global__ void marching_cubes_kernel(const r32* layer_distance1, const r32* layer_distance2,
                                      const TSDFVolume::TransformationVoxel::Ptr layer_vertices1,
                                      const TSDFVolume::TransformationVoxel::Ptr layer_vertices2,
                                      const dim3 grid_size, const float3 voxel_size,
                                      const i32 vz, float3* verts, int3* edges) {
    
    const i32 vx = threadIdx.x + blockIdx.x * blockDim.x;
    const i32 vy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if ((vx >= grid_size.x - 1) || (vy >= grid_size.y - 1))
        return;

    const ui32 voxel_idx = grid_size.x * vy + vx;
    i32 cube_idx = (grid_size.x - 1) * vy + vx;
    i32 vertex_idx = cube_idx * NUM_CUBE_SIDE_VERT;
    i32 edge_idx = cube_idx *  NUM_CUBE_TRIANGLES;
    r32 voxel_values[8] {
        layer_distance1[voxel_idx],
        layer_distance1[voxel_idx + 1],
        layer_distance2[voxel_idx + 1],
        layer_distance2[voxel_idx],
        layer_distance1[voxel_idx + grid_size.x],
        layer_distance1[voxel_idx + grid_size.x + 1],
        layer_distance2[voxel_idx + grid_size.x + 1],
        layer_distance2[voxel_idx + grid_size.x],
    };
       
    const ui8 voxel_type = define_voxel_type(voxel_values);
    if (voxel_type != 0 && voxel_type != 0xFF) {
        float3 cube_verts[8];
        compute_voxel_vertices(vx, vy, grid_size.x, layer_vertices1, layer_vertices2, cube_verts);

        float3 intersection_verts[NUM_CUBE_SIDE_VERT];
        detect_edge_intersects(voxel_type, voxel_values, cube_verts, intersection_verts);

        
        for (i32 idx = 0; idx < NUM_CUBE_SIDE_VERT; idx++) {
            if (intersection_verts[idx].x == CUDART_NAN_F)
                continue;
            verts[vertex_idx + idx] = intersection_verts[idx];
        }

        for (i32 tidx = 0, eidx = 0; tidx < NUM_CUBE_TRIANGLES; tidx++) {
            edges[edge_idx + tidx].x = MC_TRI_TABLE[voxel_type][eidx++];
            edges[edge_idx + tidx].y = MC_TRI_TABLE[voxel_type][eidx++];
            edges[edge_idx + tidx].z = MC_TRI_TABLE[voxel_type][eidx++];
        }

    } else {
        for (i32 tidx = 0; tidx < NUM_CUBE_TRIANGLES; tidx++) {
            edges[edge_idx + tidx].x = -1;
            edges[edge_idx + tidx].y = -1;
            edges[edge_idx + tidx].z = -1;
        }
    }
}

void copy_data_mesh_to_host(const TSDFVolume& tsdf_volume, const float3* cu_verts, const int3* cu_edges,
                            std::vector<float3>& vertices, std::vector<int3>& triangles) {
    const dim3 grid_size = tsdf_volume.voxel_grid_size();

    i32 cube_idx = 0;
    for (i32 y = 0; y < grid_size.y - 1; ++y) {
        for (i32 x = 0; x < grid_size.x - 1; ++x) {
            const float3* verts = cu_verts + cube_idx * NUM_CUBE_SIDE_VERT;
			const int3* edges = cu_edges + cube_idx * NUM_CUBE_TRIANGLES;
            i32 edge_idx = 0;
            while(edge_idx < NUM_CUBE_TRIANGLES && edges[edge_idx].x != -1) {
                i32 triangle_vertices[3] {edges[edge_idx].x, edges[edge_idx].y, edges[edge_idx].z};
                i32 vert_remap[NUM_CUBE_SIDE_VERT] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
                for (i32 ev = 0; ev < 3; ++ev) {
                    i32 edge_id = triangle_vertices[ev];
                    i32 v_id = vert_remap[edge_id];
                    if (v_id == -1) {
                        vertices.emplace_back(verts[edge_id]);
                        v_id = vertices.size() - 1;
                        vert_remap[edge_id] = v_id;
                    }
                    triangle_vertices[ev] = v_id;
                }

                triangles.emplace_back(make_int3(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]));
                edge_idx += 1;
            }
            cube_idx += 1;
        }
    }
}

void generate_triangular_surface(const TSDFVolume& tsdf_volume, 
                                 std::vector<float3>& vertices,
                                 std::vector<int3>& triangles) {
    const dim3 grid_size = tsdf_volume.voxel_grid_size();
    const ui32 cubes_per_layer = (grid_size.x - 1) * (grid_size.y - 1);
    const float3 voxel_size = tsdf_volume.scale_voxel_size();
    const ui32 num_verts = cubes_per_layer * NUM_CUBE_SIDE_VERT;
    float3* cu_vertices{nullptr};
    CUDA(cudaMallocManaged(&cu_vertices, num_verts * sizeof(float3)));
    
    const ui32 num_triangles = cubes_per_layer * NUM_CUBE_TRIANGLES;
    int3* cu_edges{nullptr};
    CUDA(cudaMallocManaged(&cu_edges, num_triangles * sizeof(int3)));
    
    const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(grid_size.x, blockDim.x), iDivUp(grid_size.y, blockDim.y));
    
    const float3 global_offset = tsdf_volume.global_offset();
    const r32* volume_signed_distance = tsdf_volume.voxel_distances_data();
    const TSDFVolume::TransformationVoxel::Ptr t_voxels = tsdf_volume.transformation_voxels_data(); 
    const ui32 layer_size = grid_size.x + grid_size.y;
    
    for (i32 vz = 0; vz < grid_size.z - 1; ++vz) {
        const r32* layer_distance1 = &(volume_signed_distance[vz * layer_size]);
        const r32* layer_distance2 = &(volume_signed_distance[(vz + 1) * layer_size]);
        const TSDFVolume::TransformationVoxel::Ptr layer_vertices1 = &(t_voxels[vz * layer_size]);
        const TSDFVolume::TransformationVoxel::Ptr layer_vertices2 = &(t_voxels[(vz + 1) * layer_size]);

        marching_cubes_kernel<<<gridDim, blockDim>>>(layer_distance1, layer_distance2, layer_vertices1,
                                                     layer_vertices2, grid_size, voxel_size, 
                                                     vz, cu_vertices, cu_edges);
        
        CUDA(cudaDeviceSynchronize());
        copy_data_mesh_to_host(tsdf_volume, cu_vertices, cu_edges, vertices, triangles);
    }

    CUDA(cudaDeviceSynchronize());

    if (CUDA_FAILED(cudaGetLastError())) {
        LogError(LOG_CUDA "generate_triangular_surface() -- failed extract mesh from voxel grid\n");
        return;
    }

    CUDA(cudaFree(cu_vertices));
    CUDA(cudaFree(cu_edges));
}

}