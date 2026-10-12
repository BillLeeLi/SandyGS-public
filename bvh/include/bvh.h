#ifndef BVH_BVH_H
#define BVH_BVH_H
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
create_bvh(const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& nodes, const torch::Tensor& aabbs);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities);
/*
std::tuple<torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& normals);
*/
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& normals, const torch::Tensor& scaling);


std::tuple<torch::Tensor>
accum_refl_ray_colors(const torch::Tensor& reflvec_gs_colors,
                    const torch::Tensor& reflvec_gs_alphas,
                    const torch::Tensor& reflvec_gs_Ts);

std::tuple<torch::Tensor, torch::Tensor>
get_screen_refl_colors(const torch::Tensor& rgb_fresnel,
                    const torch::Tensor& rgb_src,
                    const torch::Tensor& depth,
                    const torch::Tensor& start_coords,
                    const torch::Tensor& end_coords,
                    const torch::Tensor& start_3dpos,
                    const torch::Tensor& end_3dpos,
                    const torch::Tensor& fluid_mask,
                    const torch::Tensor& normal_fiuld_2d,
                    const float refra_k,
                    const float thickness);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_screen_reflections(const torch::Tensor& depth,
                    const torch::Tensor& normal_camera,
                    const torch::Tensor& start_coords,
                    const torch::Tensor& end_coords,
                    const torch::Tensor& start_3dpos,
                    const torch::Tensor& end_3dpos,
                    const float thickness,
                    const int32_t start_step,
                    const float frontface_cos_threshold,
                    const float backface_cos_threshold,
                    const float max_depth_jump_absolute,
                    const float max_depth_jump_relative);

#define MAX_NUM_GAUSSIANS_PER_RAY 700

#endif //BVH_BVH_H
