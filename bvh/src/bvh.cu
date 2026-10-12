#ifndef BVH_BVH_CU
#define BVH_BVH_CU
#include <tuple>
#include "bvh.h"
#include "construct.cuh"
#include "trace.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
create_bvh(const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& nodes, const torch::Tensor& aabbs){
    const uint32_t P = means3D.size(0);

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor mortons = torch::zeros({P}, means3D.options().dtype(torch::kLong));

    construct_bvh(
            P,
            means3D.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            (int32_t*)nodes.contiguous().data_ptr<int>(),
            aabbs.contiguous().data_ptr<float>(),
            (uint64_t*)mortons.contiguous().data_ptr<int64_t>()
    );
    return std::make_tuple(nodes, aabbs, mortons);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities){
    int32_t num_rays = rays_o.size(0);

    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    torch::Tensor num_contributes = torch::zeros({num_rays, 1}, int_opts);

    auto result = trace_bvh_cuda(num_rays,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   num_contributes.contiguous().data_ptr<int32_t>());

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     float milliseconds = 0;

    int32_t num_rendered = std::get<0>(result);
    thrust::device_vector<int32_t>& point_list_vec = std::get<1>(result);
    thrust::device_vector<float3>& position_list_vec = std::get<2>(result);
    thrust::device_vector<int32_t>& ray_id_list_vec = std::get<3>(result);
    if (num_rendered == 0){
        torch::Tensor point_list_tensor = torch::zeros({0, 1}, int_opts);
        torch::Tensor position_list_tensor = torch::zeros({0, 3}, float_opts);
        torch::Tensor ray_id_list_tensor = torch::zeros({0, 3}, float_opts);
        return std::make_tuple(num_contributes, point_list_tensor, position_list_tensor, ray_id_list_tensor);
    }
    int32_t* point_list = thrust::raw_pointer_cast(point_list_vec.data());
    int32_t size = point_list_vec.size();
    torch::Tensor point_list_tensor = torch::from_blob(point_list, {size, 1}, int_opts);
    point_list_tensor = point_list_tensor.clone();

    float* position_list = (float*)thrust::raw_pointer_cast(position_list_vec.data());
    torch::Tensor position_list_tensor = torch::from_blob(position_list, {size, 3}, float_opts);
    position_list_tensor = position_list_tensor.clone();

    int32_t* ray_id_list = thrust::raw_pointer_cast(ray_id_list_vec.data());
    torch::Tensor ray_id_list_tensor = torch::from_blob(ray_id_list, {size, 1}, int_opts);
    ray_id_list_tensor = ray_id_list_tensor.clone();

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "after time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);
    return std::make_tuple(num_contributes, point_list_tensor, position_list_tensor, ray_id_list_tensor);
}

/*
std::tuple<torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& normals){
    int32_t num_rays = rays_o.numel() / rays_o.size(-1);
    // printf("num_rays: %d, %d, %d\n",num_rays, rays_o.numel(), rays_o.size(-1));
    auto rays_o_shape = rays_o.sizes().slice(0, rays_o.dim() - 1);
//     auto rays_o_shape = rays_o.sizes().vec();
//     rays_o_shape.pop_back();
//     rays_o_shape.push_back(1);

    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    torch::Tensor num_contributes = torch::zeros(rays_o_shape, int_opts);
    torch::Tensor rendered_opacity = torch::ones(rays_o_shape, float_opts);

    trace_bvh_opacity_cuda(num_rays,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   (float3*)normals.contiguous().data_ptr<float>(),
                   num_contributes.contiguous().data_ptr<int32_t>(),
                   rendered_opacity.contiguous().data_ptr<float>());
    return std::make_tuple(num_contributes, rendered_opacity);
}
*/


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& normals, const torch::Tensor& scaling){
    int32_t num_rays = rays_o.numel() / rays_o.size(-1);
    // printf("num_rays: %d, %d, %d\n",num_rays, rays_o.numel(), rays_o.size(-1));
    auto rays_o_shape = rays_o.sizes().slice(0, rays_o.dim() - 1);
    //printf("rays_o_shape: %d, %d", rays_o_shape[0], rays_o_shape[1]);
    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    torch::Tensor reflvec_gs_ids = torch::full({num_rays, MAX_NUM_GAUSSIANS_PER_RAY}, -1, int_opts);
    torch::Tensor reflvec_gs_alphas = torch::full({num_rays, MAX_NUM_GAUSSIANS_PER_RAY}, -1, float_opts);
    torch::Tensor reflvec_gs_Ts = torch::full({num_rays, MAX_NUM_GAUSSIANS_PER_RAY}, -1, float_opts);

    trace_bvh_opacity_cuda(num_rays,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   (float3*)normals.contiguous().data_ptr<float>(),
                   (float3*)scaling.contiguous().data_ptr<float>(),
                   reflvec_gs_ids.contiguous().data_ptr<int32_t>(),
                   reflvec_gs_alphas.contiguous().data_ptr<float>(),
                   reflvec_gs_Ts.contiguous().data_ptr<float>());
    return std::make_tuple(reflvec_gs_ids, reflvec_gs_alphas, reflvec_gs_Ts);
}


std::tuple<torch::Tensor>
accum_refl_ray_colors(const torch::Tensor& reflvec_gs_colors,
                    const torch::Tensor& reflvec_gs_alphas,
                    const torch::Tensor& reflvec_gs_Ts){
    int32_t num_rays = reflvec_gs_colors.size(0);
    printf("num_rays: %d\n", num_rays);
    int32_t reflvec_gs_ids_size1 = MAX_NUM_GAUSSIANS_PER_RAY;
    auto int_opts = reflvec_gs_colors.options().dtype(torch::kInt32);
    auto float_opts = reflvec_gs_colors.options();

    torch::Tensor refl_ray_colors = torch::full({num_rays, 3}, 0, float_opts);

    accum_refl_rays_color_cuda(num_rays,
                   reflvec_gs_colors.contiguous().data_ptr<float>(),
                   reflvec_gs_alphas.contiguous().data_ptr<float>(),
                   reflvec_gs_Ts.contiguous().data_ptr<float>(),
                   (float3*)refl_ray_colors.contiguous().data_ptr<float>(),
                   reflvec_gs_ids_size1);
    return std::make_tuple(refl_ray_colors);
}


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
                    const float thickness){
    int32_t num_pixels = start_coords.size(0);
    printf("num_pixels: %d\n", num_pixels);

    auto float_opts = rgb_fresnel.options();
    torch::Tensor screen_refl_colors = torch::full({num_pixels, 3}, -1, float_opts);
    torch::Tensor screen_refra_colors = torch::full({num_pixels, 3}, -1, float_opts);
    //auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    //torch::Tensor nohit_indices = torch::full({num_pixels}, 0, int_opts);

    int32_t img_height = rgb_fresnel.size(0);
    int32_t img_width = rgb_fresnel.size(1);

    get_screen_refl_color_cuda(num_pixels,
                   img_height,
                   img_width,
                   rgb_fresnel.contiguous().data_ptr<float>(),
                   rgb_src.contiguous().data_ptr<float>(),
                   depth.contiguous().data_ptr<float>(),
                   start_coords.contiguous().data_ptr<float>(),
                   end_coords.contiguous().data_ptr<float>(),
                   start_3dpos.contiguous().data_ptr<float>(),
                   end_3dpos.contiguous().data_ptr<float>(),
                   (float3*)screen_refl_colors.contiguous().data_ptr<float>(),
                   fluid_mask.contiguous().data_ptr<float>(),
                   normal_fiuld_2d.contiguous().data_ptr<float>(),
                   (float3*)screen_refra_colors.contiguous().data_ptr<float>(),
                   refra_k,
                   thickness
                   );
    return std::make_tuple(screen_refl_colors, screen_refra_colors);
}

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
                    const float max_depth_jump_relative) {
    const int32_t num_pixels = start_coords.size(0);
    const int32_t img_height = depth.size(0);
    const int32_t img_width = depth.size(1);
    auto int_opts = depth.options().dtype(torch::kInt32);
    torch::Tensor hit_coords = torch::full({num_pixels, 2}, -1, int_opts);
    torch::Tensor hit_status = torch::zeros({num_pixels}, int_opts);
    torch::Tensor proxy_coords = torch::full({num_pixels, 2}, -1, int_opts);
    torch::Tensor hit_normal_dot = torch::zeros({num_pixels}, depth.options());
    trace_screen_reflection_cuda(
        num_pixels, img_height, img_width,
        depth.contiguous().data_ptr<float>(),
        normal_camera.contiguous().data_ptr<float>(),
        start_coords.contiguous().data_ptr<float>(),
        end_coords.contiguous().data_ptr<float>(),
        start_3dpos.contiguous().data_ptr<float>(),
        end_3dpos.contiguous().data_ptr<float>(),
        hit_coords.contiguous().data_ptr<int32_t>(),
        hit_status.contiguous().data_ptr<int32_t>(),
        proxy_coords.contiguous().data_ptr<int32_t>(),
        hit_normal_dot.contiguous().data_ptr<float>(),
        thickness, start_step, frontface_cos_threshold,
        backface_cos_threshold, max_depth_jump_absolute,
        max_depth_jump_relative);
    return std::make_tuple(hit_coords, hit_status, proxy_coords, hit_normal_dot);
}


#endif //BVH_BVH_CU
