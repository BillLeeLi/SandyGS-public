#ifndef BVH_TRACE_CUH
#define BVH_TRACE_CUH
#include <cstdint>
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include "utility.cuh"

template <typename T, int MAX_SIZE=32>
class IndexStack {
public:
    __host__ __device__ void push(T val) {
        if (m_count >= MAX_SIZE - 1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

std::tuple<int32_t, thrust::device_vector<int32_t>, thrust::device_vector<float3>, thrust::device_vector<int32_t>>
trace_bvh_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, int32_t* contributes);
/*
void trace_bvh_opacity_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    int32_t* contributes,
                    float* rendered_opacity);
*/
void trace_bvh_opacity_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    float3* scaling,
                    int32_t* reflvec_gs_ids,
                    float* reflvec_gs_alphas,
                    float* reflvec_gs_Ts);

void accum_refl_rays_color_cuda(int32_t num_rays,
                    float* reflvec_gs_colors,
                    float* reflvec_gs_alphas,
                    float* reflvec_gs_Ts,
                    float3* refl_ray_colors,
                    int32_t reflvec_gs_ids_size1);

void get_screen_refl_color_cuda(int32_t num_pixels,
                    int32_t img_height,
                    int32_t img_width,
                    float* rgb_fresnel,
                    float* rgb_src,
                    float* depth,
                    float* start_coords,
                    float* end_coords,
                    float* start_3dpos,
                    float* end_3dpos,
                    float3* screen_refl_colors,
                    float* fluid_mask,
                    float* normal_fiuld_2d,
                    float3* screen_refra_colors,
                    float refra_k,
                    float thickness);

// P4 returns ray/geometry intersections to the optical pass. This allows
// tracing against composed source+sand depth while applying per-pixel water
// thickness during Python-side shading.
void trace_screen_reflection_cuda(int32_t num_pixels,
                    int32_t img_height,
                    int32_t img_width,
                    float* depth,
                    float* normal_camera,
                    float* start_coords,
                    float* end_coords,
                    float* start_3dpos,
                    float* end_3dpos,
                    int32_t* hit_coords,
                    int32_t* hit_status,
                    int32_t* proxy_coords,
                    float* hit_normal_dot,
                    float thickness,
                    int32_t start_step,
                    float frontface_cos_threshold,
                    float backface_cos_threshold,
                    float max_depth_jump_absolute,
                    float max_depth_jump_relative);


#endif //BVH_TRACE_CUH
