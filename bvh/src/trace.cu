#include "trace.cuh"
#include "bvh.h"

struct id_intersection{
    int32_t id;
    float2 intersection;
};

std::tuple<int32_t, thrust::device_vector<int32_t>, thrust::device_vector<float3>, thrust::device_vector<int32_t>>
trace_bvh_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, int32_t* num_contributes){
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        float milliseconds = 0;


        auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);

        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d, num_contributes] __device__ (int32_t idx)
                     {
                         IndexStack<int32_t> stack_device;
                         stack_device.push(0);
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         while (!stack_device.empty()) {
                             int32_t node_id = stack_device.pop();
                             int32_t* node = nodes + node_id*5;
                             if (node[4] <= 4){
                                 count += node[4];
                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                 }
                             }
                         }
                         num_contributes[idx] = count;
                     });

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "num_contributes time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);

    thrust::device_vector<int32_t> ray_offsets_vec(num_rays);
    thrust::device_ptr<int32_t> num_contributes_ptr = thrust::device_pointer_cast(num_contributes);

    thrust::inclusive_scan(num_contributes_ptr, num_contributes_ptr+num_rays, ray_offsets_vec.begin());
    int32_t* ray_offsets = thrust::raw_pointer_cast(ray_offsets_vec.data());
    int32_t num_rendered;
    cudaMemcpy(&num_rendered, ray_offsets + num_rays - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    // printf("num_rendered: %d\n",num_rendered);

    thrust::device_vector<uint64_t> ray_list_key_vec(num_rendered);
    uint64_t* ray_list_key = thrust::raw_pointer_cast(ray_list_key_vec.data());

    thrust::device_vector<int32_t> point_list_vec(num_rendered);
    int32_t* point_list = thrust::raw_pointer_cast(point_list_vec.data());
    thrust::device_vector<float3> position_list_vec(num_rendered);
    float3* position_list = thrust::raw_pointer_cast(position_list_vec.data());
    thrust::device_vector<int32_t> ray_id_list_vec(num_rendered);
    int32_t* ray_id_list = thrust::raw_pointer_cast(ray_id_list_vec.data());

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "inclusive_scan time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);

    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<uint32_t>(0),
                     thrust::make_counting_iterator<uint32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d, num_contributes,
                      ray_offsets, ray_list_key, point_list, ray_id_list,
                             means3D, covs3D, position_list
                     ] __device__ (uint32_t idx)
                     {
                         if (num_contributes[idx]==0) return;
                         uint32_t offset = (idx == 0) ? 0 : ray_offsets[idx - 1];
                         uint64_t* ray_key_ptr=ray_list_key+offset;
                         int32_t* point_ptr=point_list+offset;
                         float3* position_ptr=position_list+offset;
                         int32_t* ray_id_ptr=ray_id_list+offset;

                         IndexStack<id_intersection> stack_device;
                         stack_device.push({0, {-1000, 1000}});
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         while (!stack_device.empty()) {
                             id_intersection pop_result = stack_device.pop();
                             int32_t node_id = pop_result.id;
                             float2 intersection = pop_result.intersection;
                             int32_t* node = nodes + node_id*5;
                             if (node[4] <= 4){
                                 IndexStack<int32_t> stack2;
                                 stack2.push(node_id);
                                 int32_t count2 = 0;
                                 int32_t count2_total = node[4];
                                 while (!stack2.empty()) {
                                     int32_t node_id = stack2.pop();
                                     int32_t* node = nodes + node_id*5;
                                     if (node[3] >= 0){

                                         int32_t object_id = node[3];
                                         float t = ray_intersects(means3D[object_id], ray_o, ray_d);
                                         // float t = ray_intersects(means3D[object_id], covs3D + object_id*6, ray_o, ray_d);
                                         if (t<0.01 || t < intersection.x || t > intersection.y){
                                            t = 1000000.f;
                                            object_id = -1;
                                         }
                                         uint64_t key = idx;
                                         key <<= 32;
                                         key |= *(uint32_t*)&t;
                                         ray_key_ptr[count] = key;
                                         point_ptr[count] = object_id;
                                         ray_id_ptr[count] = idx;
                                         position_ptr[count] = {
                                                 ray_o.x + t * ray_d.x,
                                                 ray_o.y + t * ray_d.y,
                                                 ray_o.z + t * ray_d.z,
                                         };
                                         ++count2;
                                         ++count;
                                     }else{
                                         stack2.push(node[1]);
                                         stack2.push(node[2]);
                                     }
                                 }
                                 assert(count2 == count2_total);



                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push({lid, interection_l});
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push({rid, interection_r});
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push({rid, interection_r});
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push({lid, interection_l});
                                     }
                                 }
                             }
                         }
                     });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "trace time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);
    thrust::stable_sort_by_key(ray_list_key_vec.begin(),
                               ray_list_key_vec.end(),
                               thrust::make_zip_iterator(
                                       thrust::make_tuple(point_list_vec.begin(),
                                                          position_list_vec.begin())));
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "sort time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);
    return std::make_tuple(num_rendered, point_list_vec, position_list_vec, ray_id_list_vec);
}

/*
//__device__ int max_count_device = 0;
void trace_bvh_opacity_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    int32_t* num_contributes,
                    float* rendered_opacity){
//         cudaEvent_t start, stop;
//         cudaEventCreate(&start);
//         cudaEventCreate(&stop);
//         cudaEventRecord(start);
//         float milliseconds = 0;
        auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);

        //int max_count_host = 0;
        //cudaMemcpyToSymbol(max_count_device, &max_count_host, sizeof(int));

        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d, num_contributes,
                     means3D, covs3D, opacities, normals, rendered_opacity] __device__ (int32_t idx)
                     {
                         IndexStack<int32_t> stack_device;
                         stack_device.push(0);
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         // debug
                         //if (idx == 0) {
                         //   printf("idx: %d, x=%.2f, y=%.2f, z=%.2f\n", idx, ray_o.x, ray_o.y, ray_o.z);}
                         //
                         float ray_opacity = 1.0f;
                         while (!stack_device.empty()) {
                             int32_t node_id = stack_device.pop();
                             int32_t* node = nodes + node_id*5;
                             // 如果当前节点是叶子节点（或接近叶子节点）
                             if (node[4] <= 1){  // 8
                                 IndexStack<int32_t> stack_device2;
                                 stack_device2.push(node_id);
                                 while (!stack_device2.empty()) {
                                    int32_t node_id2 = stack_device2.pop();
                                    int32_t* node2 = nodes + node_id2*5;
                                    // 如果当前节点不是叶子节点，将其子节点压入栈中；否则，处理叶子节点。
                                    if (node2[4] > 1){
                                        stack_device2.push(node2[1]);
                                        stack_device2.push(node2[2]);
                                    }else{
                                        int32_t object_id2 = node2[3];
                                        if (opacities[object_id2]<1.f/255.f) continue;
                                        float3 normal = normals[object_id2];
                                        // 光线与法线的点积大于零，表示光线与表面之间的角度小于90°，即光线从表面外部射入表面，照亮了它，光线对表面是有贡献的，continue
                                        if (normal.x*ray_d.x+normal.y*ray_d.y+normal.z*ray_d.z>0) continue;
                                        // Eq.14
                                        float t = ray_intersects(means3D[object_id2], covs3D + object_id2*6, ray_o, ray_d);
                                         if (t<0.01){
                                            continue;
                                         }
                                         // the equivalent intersection point
                                         float3 pos = {
                                                         ray_o.x + t * ray_d.x,
                                                         ray_o.y + t * ray_d.y,
                                                         ray_o.z + t * ray_d.z,
                                         };
                                         // 在给定均值和协方差矩阵的情况下，计算pos位置的多元高斯概率密度函数的指数部分
                                         float power = gaussian_fn(means3D[object_id2], pos, covs3D + object_id2*6);
                                         if (power>0) continue;
                                         count += 1;
                                         // 计算pos位置的alpha值
                                         float alpha = opacities[object_id2] * __expf(power);
                                         // the transmittance T is progressively attenuated
                                         ray_opacity *= 1-alpha;
                                         if (idx == 0) {
                                             printf("idx: %d, x=%.2f, y=%.2f, z=%.2f, ray_opacity=%.4f, count=%d\n", idx, pos.x, pos.y, pos.z, ray_opacity, count);}
                                         // early terminated if a ray’s transmittance drops below a certain threshold
                                         if (ray_opacity<0.1) {
                                            rendered_opacity[idx] = 0.0f;
                                            return;
                                         }
                                    }
                                 }
                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                 }
                             }
                         }
                         num_contributes[idx] = count;
                         rendered_opacity[idx] = ray_opacity;

                         // 更新最大 count
                         // atomicMax(&max_count_device, count);
                     });

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "tracing time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);

// 同步设备
// cudaDeviceSynchronize();
// 获取设备上的最大 count
// cudaMemcpyFromSymbol(&max_count_host, max_count_device, sizeof(int));
// std::cout << "Maximum count: " << max_count_host << std::endl;
}
*/

struct IntersectionData {
    float t;
    int32_t reflvec_gs_id;
    float reflvec_gs_alpha;
    //float reflvec_gs_T;
};

__device__ void merge(IntersectionData* data, IntersectionData* temp, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (data[i].t <= data[j].t) {
            temp[k++] = data[i++];
        } else {
            temp[k++] = data[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = data[i++];
    }

    while (j <= right) {
        temp[k++] = data[j++];
    }

    for (i = left; i <= right; i++) {
        data[i] = temp[i];
    }
}

__device__ void merge_sort(IntersectionData* data, IntersectionData* temp, int size) {
    for (int width = 1; width < size; width *= 2) {
        for (int i = 0; i < size; i += 2 * width) {
            int left = i;
            int mid = min(i + width - 1, size - 1);
            int right = min(i + 2 * width - 1, size - 1);
            if (mid < right) {
                merge(data, temp, left, mid, right);
            }
        }
    }
}

template<int max_num_gaussians_per_ray>
void trace_bvh_opacity_cuda_0(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    float3* scaling,
                    int32_t* reflvec_gs_ids,
                    float* reflvec_gs_alphas,
                    float* reflvec_gs_Ts){
        cudaDeviceSetLimit(cudaLimitStackSize, 16384); // 增加设备栈大小
        auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);

        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_rays),
                     [nodes, aabbs_internal, rays_o, rays_d,
                     means3D, covs3D, opacities, normals, scaling, reflvec_gs_ids, reflvec_gs_alphas, reflvec_gs_Ts, num_rays] __device__ (int32_t idx)
                     {
                         IndexStack<int32_t> stack_device;
                         stack_device.push(0);
                         int32_t count = 0;
                         float3 ray_o=rays_o[idx],ray_d=rays_d[idx];
                         float ray_opacity = 1.0f;

                         ///
                         IntersectionData intersections[max_num_gaussians_per_ray]; // 假设最多记录 max_num_gaussians_per_ray 个交点
                         bool early_terminated = false; /// 标记是否提前终止
                         //printf("current ray idx %d\n", idx);
                         ///

                         while (!stack_device.empty()) {
                             int32_t node_id = stack_device.pop();
                             int32_t* node = nodes + node_id*5;
                             // 如果当前节点是叶子节点（或接近叶子节点）
                             if (node[4] <= 1){  // 8
                                 IndexStack<int32_t> stack_device2;
                                 stack_device2.push(node_id);
                                 while (!stack_device2.empty()) {
                                    int32_t node_id2 = stack_device2.pop();
                                    int32_t* node2 = nodes + node_id2*5;
                                    // 如果当前节点不是叶子节点，将其子节点压入栈中；否则，处理叶子节点。
                                    if (node2[4] > 1){
                                        stack_device2.push(node2[1]);
                                        stack_device2.push(node2[2]);
                                    }else{
                                        int32_t object_id2 = node2[3];
                                        ///
                                        if (scaling[object_id2].x>50 || scaling[object_id2].y>50 || scaling[object_id2].z>50) continue;
                                        ///
                                        if (opacities[object_id2]<1.f/255.f) continue;

                                        /*
                                        // 光线与法线的点积大于零，表示光线与表面之间的角度小于90°，即光线从表面外部射入表面，照亮了它，光线对表面是有贡献的，continue
                                        float3 normal = normals[object_id2];
                                        if (normal.x*ray_d.x+normal.y*ray_d.y+normal.z*ray_d.z>0) continue;
                                        */

                                         // Eq.14
                                         float t = ray_intersects(means3D[object_id2], covs3D + object_id2*6, ray_o, ray_d);
                                         if (t < 0.01){
                                            continue;
                                         }

                                         // the equivalent intersection point
                                         float3 pos = {
                                                         ray_o.x + t * ray_d.x,
                                                         ray_o.y + t * ray_d.y,
                                                         ray_o.z + t * ray_d.z,
                                         };

                                         // 在给定均值和协方差矩阵的情况下，计算pos位置的多元高斯概率密度函数的指数部分
                                         float power = gaussian_fn(means3D[object_id2], pos, covs3D + object_id2*6);
                                         if (power > 0) continue;

                                         // 计算pos位置的alpha值
                                         float alpha = opacities[object_id2] * __expf(power);

                                         // the transmittance T is progressively attenuated
                                         ray_opacity *= 1-alpha;

                                         ///
                                         // 不透明度太低则不记录
                                         if (alpha < 10.0f / 255.0f)  continue;

                                         // 记录反射光路上的gaussian infos
                                         //if (idx == 44152) {
                                         //    printf("Before sort: idx %d, t %f, object_id2 %d, count %d, alpha %f\n", idx, t, object_id2, count, alpha);}
                                         /*
                                         if (count < max_num_gaussians_per_ray){
                                            intersections[count] = {t, object_id2, alpha, ray_opacity};
                                            count += 1;
                                         } else {
                                            early_terminated = true;
                                            break;
                                         }
                                         */

                                         if (count >= max_num_gaussians_per_ray) printf("count: %d %d", count, max_num_gaussians_per_ray);
                                         intersections[count] = {t, object_id2, alpha};
                                         count += 1;

                                         /*
                                         // early terminated if a ray’s transmittance drops below a certain threshold
                                         if (ray_opacity<0.001) {
                                            //return;
                                            early_terminated = true;
                                            break;
                                         }
                                         */
                                    }
                                 }
                                 ///
                                 if (early_terminated) break;
                                 ///
                             }else{
                                 int32_t lid = node[1], rid = node[2];
                                 float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                                 float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                                 if (interection_l.y > interection_r.y){
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                 }else{
                                     if (interection_r.y>0){
                                         stack_device.push(rid);
                                     }
                                     if (interection_l.y>0){
                                         stack_device.push(lid);
                                     }
                                 }
                             }
                         }

                         IntersectionData temp[max_num_gaussians_per_ray];
                         if (count > 0)
                            merge_sort(intersections, temp, count);

                         float new_ray_opacity = 1.0f;
                         for (int i = 0; i < count; ++i) {
                             int32_t index = idx * max_num_gaussians_per_ray + i;

                             float cur_alpha = intersections[i].reflvec_gs_alpha;
                             new_ray_opacity *= 1-cur_alpha;
                             if (new_ray_opacity < 0.01) break;

                             if (index >= max_num_gaussians_per_ray * num_rays)
                                 printf("%d %d %d %d %d %f %f\n", index, idx, max_num_gaussians_per_ray, i, intersections[i].reflvec_gs_id, cur_alpha, new_ray_opacity);

                             reflvec_gs_ids[index] = intersections[i].reflvec_gs_id;
                             reflvec_gs_alphas[index] = cur_alpha;
                             reflvec_gs_Ts[index] = new_ray_opacity;

                             //if (idx == 44152) {
                             //    printf("After sort: idx %d, t %f, gs_id %d, alpha %f, T %f\n", idx, intersections[i].t, reflvec_gs_ids[index], reflvec_gs_alphas[index], reflvec_gs_Ts[index]);}
                         }

                     });
cudaDeviceSynchronize();
}

void trace_bvh_opacity_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
                    float3* rays_o, float3* rays_d,
                    float3* means3D, float* covs3D,
                    float* opacities, float3* normals,
                    float3* scaling,
                    int32_t* reflvec_gs_ids,
                    float* reflvec_gs_alphas,
                    float* reflvec_gs_Ts){
                        trace_bvh_opacity_cuda_0<MAX_NUM_GAUSSIANS_PER_RAY>(
                            num_rays, nodes, aabbs,
                            rays_o, rays_d,
                            means3D, covs3D,
                            opacities, normals,
                            scaling,
                            reflvec_gs_ids,
                            reflvec_gs_alphas,
                            reflvec_gs_Ts
                        );
                    }

void accum_refl_rays_color_cuda(int32_t num_rays,
                    float* reflvec_gs_colors,
                    float* reflvec_gs_alphas,
                    float* reflvec_gs_Ts,
                    float3* refl_ray_colors,
                    int32_t reflvec_gs_ids_size1){
        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_rays),
                     [reflvec_gs_colors, reflvec_gs_alphas, reflvec_gs_Ts, refl_ray_colors, reflvec_gs_ids_size1] __device__ (int32_t ray_idx)
                     {
                         for (int gaussian_idx=0; gaussian_idx<reflvec_gs_ids_size1; gaussian_idx++){

                             int query_index = ray_idx * reflvec_gs_ids_size1 + gaussian_idx;

                             float gaussian_color_r = reflvec_gs_colors[query_index * 3 + 0];
                             float gaussian_color_g = reflvec_gs_colors[query_index * 3 + 1];
                             float gaussian_color_b = reflvec_gs_colors[query_index * 3 + 2];
                             if (gaussian_color_r == -1.000f && gaussian_color_g == -1.000f && gaussian_color_b == -1.000f) {
                                continue;
                             }

                             float bg_color[3] = {0.0f, 0.0f, 0.0f};
                             float alpha = reflvec_gs_alphas[query_index];
                             float T = reflvec_gs_Ts[query_index];
                             refl_ray_colors[ray_idx].x += gaussian_color_r * alpha * T; // + T * bg_color[0];
                             refl_ray_colors[ray_idx].y += gaussian_color_g * alpha * T; //+ T * bg_color[1];
                             refl_ray_colors[ray_idx].z += gaussian_color_b * alpha * T; //+ T * bg_color[2];

                             //if (ray_idx == 0) {
                             //   printf("#######2 %f %f %f, %f %f %f\n", refl_ray_colors[ray_idx].x, refl_ray_colors[ray_idx].y, refl_ray_colors[ray_idx].z, gaussian_color_r, alpha, T);
                             //}
                         }
                     });
cudaDeviceSynchronize();
}


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
                                float thickness){
        thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<int32_t>(0),
                     thrust::make_counting_iterator<int32_t>(num_pixels),
                     [img_height, img_width, rgb_fresnel, rgb_src, depth, start_coords, end_coords, start_3dpos, end_3dpos, screen_refl_colors, fluid_mask, normal_fiuld_2d, screen_refra_colors, refra_k, thickness] __device__ (int32_t pixel_idx)
                     {
                         // 提取起始/终止的2D/3D坐标
                         float start_x = start_coords[2 * pixel_idx];
                         float start_y = start_coords[2 * pixel_idx + 1];
                         float end_x = end_coords[2 * pixel_idx];
                         float end_y = end_coords[2 * pixel_idx + 1];
                         float dx = end_x - start_x;
                         float dy = end_y - start_y;

                         float start_x_3d = start_3dpos[3 * pixel_idx];
                         float start_y_3d = start_3dpos[3 * pixel_idx + 1];
                         float start_z_3d = start_3dpos[3 * pixel_idx + 2];
                         float end_x_3d = end_3dpos[3 * pixel_idx];
                         float end_y_3d = end_3dpos[3 * pixel_idx + 1];
                         float end_z_3d = end_3dpos[3 * pixel_idx + 2];

                         //if (dx < 0 || dy < 0){
                         //   printf("%d %f %f\n", pixel_idx, dx, dy);
                         //}
                         //if (pixel_idx == 596449){
                         //   printf("start_x, start_y, end_x, end_y: %f %f %f %f\n", start_x, start_y, end_x, end_y);
                         //   printf("start_x_3d, start_y_3d, start_z_3d, end_x_3d, end_y_3d, end_z_3d: %f %f %f %f %f %f\n", start_x_3d, start_y_3d, start_z_3d, end_x_3d, end_y_3d, end_z_3d);
                         //}

                        // 计算增量步长
                        int max_steps = static_cast<int>(fmaxf(fabsf(dx), fabsf(dy)));
                        float x_inc = dx / max_steps;
                        float y_inc = dy / max_steps;

                        // 初始化当前坐标
                        float x = start_x;
                        float y = start_y;

                        //int hit = 0; // 用于标记是否击中

                        for (int step = 0; step < max_steps; ++step) {
                            int x_rounded = static_cast<int>(roundf(x));
                            int y_rounded = static_cast<int>(roundf(y));

                            // 检查是否在图像边界内
                            if (x_rounded >= 0 && x_rounded < img_width && y_rounded >= 0 && y_rounded < img_height) {
                                // calculate t & s
                                float t = static_cast<float>(step) / static_cast<float>(max_steps);
                                float s = (start_z_3d * t) / (end_z_3d * (1-t) + start_z_3d * t);

                                // calculate x_cor, y_cor, z_cor
                                float x_cor = start_x_3d * (1-s) + end_z_3d * s;
                                float y_cor = start_y_3d * (1-s) + end_y_3d * s;
                                float z_cor = start_z_3d * (1-s) + end_z_3d * s;

                                // calculate distance
                                //float distance_i = sqrtf(x_cor * x_cor + y_cor * y_cor + z_cor * z_cor);
                                float distance_i = z_cor;

                                // 获取深度图上的深度
                                int32_t index = y_rounded * img_width + x_rounded;
                                float scene_depth_i = depth[index * 3];

                                //if (pixel_idx == 771874){
                                //     printf("%d: %d %d %f %f %f %d\n", step, x_rounded, y_rounded, scene_depth_i, distance_i, t, max_steps);
                                //}

                                // 相交测试
                                //float thickness = 0.03;
                                //float thickness = 0.01 * scene_depth_i;
                                if (distance_i > scene_depth_i && distance_i < (scene_depth_i + thickness)) {
                                   /*
                                   // 计算距离屏幕边界的最近距离
                                   int d_left = x_rounded;
                                   int d_right = img_width - x_rounded;
                                   int d_top = y_rounded;
                                   int d_bottom = img_height - y_rounded;
                                   int d_edge = fminf(fminf(d_left, d_right), fminf(d_top, d_bottom));

                                   // 计算衰减因子，线性衰减
                                   float att_thres = 0.1;
                                   float attenuation = fminf(fmaxf(d_edge / att_thres, 0.0f), 1.0f);
                                   // 指数衰减: float attenuation = exp(-k * (att_thres - d_edge));
                                   */
                                   float attenuation = 1.0f;
                                   screen_refl_colors[pixel_idx].x = rgb_fresnel[index * 3] * attenuation;
                                   screen_refl_colors[pixel_idx].y = rgb_fresnel[index * 3 + 1] * attenuation;
                                   screen_refl_colors[pixel_idx].z = rgb_fresnel[index * 3 + 2] * attenuation;

                                   // 标记为击中
                                   //hit = 1;
                                   // 结束循环！
                                   break;
                                }
                            }

                            // 如果超出边界，则停止
                            if (x < 0 || x >= img_width || y < 0 || y >= img_height) {
                                break;
                            }

                            // 更新坐标
                            x += x_inc;
                            y += y_inc;
                        }

                        // 如果没有击中，标记 nohit_indices 为 1
                        //if (hit == 0) {
                        //    nohit_indices[pixel_idx] = 1;
                        //}


                        // 计算折射颜色
                        float normal_2d_x = normal_fiuld_2d[2 * pixel_idx];
                        float normal_2d_y = normal_fiuld_2d[2 * pixel_idx + 1];

                        float moved_x = start_x + normal_2d_x * refra_k;
                        float moved_y = start_y + normal_2d_y * refra_k;

                        int start_x_rounded = static_cast<int>(roundf(start_x));
                        int start_y_rounded = static_cast<int>(roundf(start_y));
                        int moved_x_rounded = static_cast<int>(roundf(moved_x));
                        int moved_y_rounded = static_cast<int>(roundf(moved_y));

                        //if (pixel_idx == 193988)
                        //    printf("%d, %d, %d, %d\n", start_x_rounded, start_y_rounded, moved_x_rounded, moved_y_rounded);

                        int index_move_before = start_y_rounded * img_width + start_x_rounded;
                        int index_move_after = moved_y_rounded * img_width + moved_x_rounded;

                        if (moved_x_rounded >= 0 && moved_x_rounded < img_width && moved_y_rounded >= 0 && moved_y_rounded < img_height) {
                            //printf("1 pixel_idx: %d\n", pixel_idx);
                            if (fluid_mask[index_move_after * 3] == 1.0f || fluid_mask[index_move_after * 3] == 2.0f) {
                                //printf("pixel_idx: %d\n", pixel_idx);

                                screen_refra_colors[pixel_idx].x = rgb_src[index_move_after * 3];
                                screen_refra_colors[pixel_idx].y = rgb_src[index_move_after * 3 + 1];
                                screen_refra_colors[pixel_idx].z = rgb_src[index_move_after * 3 + 2];
                            } else {
                                screen_refra_colors[pixel_idx].x = rgb_src[index_move_before * 3];
                                screen_refra_colors[pixel_idx].y = rgb_src[index_move_before * 3 + 1];
                                screen_refra_colors[pixel_idx].z = rgb_src[index_move_before * 3 + 2];
                            }
                        } else {
                            screen_refra_colors[pixel_idx].x = rgb_src[index_move_before * 3];
                            screen_refra_colors[pixel_idx].y = rgb_src[index_move_before * 3 + 1];
                            screen_refra_colors[pixel_idx].z = rgb_src[index_move_before * 3 + 2];
                        }

                     });
cudaDeviceSynchronize();
}

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
                                float max_depth_jump_relative) {
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(num_pixels),
        [img_height, img_width, depth, normal_camera, start_coords, end_coords,
         start_3dpos, end_3dpos, hit_coords, hit_status, proxy_coords,
         hit_normal_dot, thickness, start_step, frontface_cos_threshold,
         backface_cos_threshold, max_depth_jump_absolute,
         max_depth_jump_relative] __device__ (int32_t pixel_idx) {
            const float start_x = start_coords[2 * pixel_idx];
            const float start_y = start_coords[2 * pixel_idx + 1];
            const float end_x = end_coords[2 * pixel_idx];
            const float end_y = end_coords[2 * pixel_idx + 1];
            const float dx = end_x - start_x;
            const float dy = end_y - start_y;
            const int max_steps = static_cast<int>(fmaxf(fabsf(dx), fabsf(dy)));
            if (max_steps <= 0) return;

            const float start_z = start_3dpos[3 * pixel_idx + 2];
            const float end_z = end_3dpos[3 * pixel_idx + 2];
            const float ray_dx = end_3dpos[3 * pixel_idx] -
                                 start_3dpos[3 * pixel_idx];
            const float ray_dy = end_3dpos[3 * pixel_idx + 1] -
                                 start_3dpos[3 * pixel_idx + 1];
            const float ray_dz = end_z - start_z;
            const float ray_norm = sqrtf(
                ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz
            );
            if (ray_norm <= 1.0e-8f) return;
            const float inv_ray_norm = 1.0f / ray_norm;

            // Seed the signed crossing test at the water pixel. The composed
            // opaque substrate should be behind the water, hence delta < 0.
            // Requiring a negative-to-positive transition prevents a ray that
            // is already behind geometry from being accepted as a new hit.
            const int start_xi = static_cast<int>(roundf(start_x));
            const int start_yi = static_cast<int>(roundf(start_y));
            bool has_previous = false;
            float previous_delta = 0.0f;
            float previous_scene_z = 0.0f;
            if (start_xi >= 0 && start_xi < img_width &&
                start_yi >= 0 && start_yi < img_height) {
                const float initial_scene_z = depth[start_yi * img_width + start_xi];
                if (isfinite(initial_scene_z) && initial_scene_z > 0.0f) {
                    previous_delta = start_z - initial_scene_z;
                    previous_scene_z = initial_scene_z;
                    has_previous = true;
                }
            }

            bool saw_backface = false;
            bool saw_ambiguous = false;
            const int first_step = max(1, start_step);
            for (int step = first_step; step < max_steps; ++step) {
                const float t = static_cast<float>(step) /
                                static_cast<float>(max_steps);
                const float x = start_x + dx * t;
                const float y = start_y + dy * t;
                const int xi = static_cast<int>(roundf(x));
                const int yi = static_cast<int>(roundf(y));
                if (xi < 0 || xi >= img_width || yi < 0 || yi >= img_height)
                    break;

                // Use perspective-correct interpolation for projected ray samples.
                // Skipping the first pixels avoids self-intersection with the
                // water plane; only the composed opaque depth is supplied here.
                const float denominator = end_z * (1.0f - t) + start_z * t;
                if (fabsf(denominator) < 1.0e-8f) continue;
                const float s = (start_z * t) / denominator;
                const float ray_z = start_z * (1.0f - s) + end_z * s;
                const float scene_z = depth[yi * img_width + xi];
                if (!isfinite(scene_z) || scene_z <= 0.0f) {
                    has_previous = false;
                    continue;
                }

                const float delta = ray_z - scene_z;
                // A signed negative-to-positive transition is the primary
                // intersection test. Do not cap its overshoot: a one-pixel DDA
                // step can cross a larger camera-depth interval on a slanted or
                // slightly noisy 3DGS surface. If the ray traversed a depth hole,
                // retain the legacy thickness only for a near-surface restart.
                const bool crossed_surface = has_previous &&
                    previous_delta < -1.0e-5f && delta >= 0.0f;
                const bool restarted_near_surface = !has_previous &&
                    delta >= 0.0f && delta < thickness;
                if (crossed_surface || restarted_near_surface) {
                    const int normal_offset = 3 * (yi * img_width + xi);
                    const float nx = normal_camera[normal_offset];
                    const float ny = normal_camera[normal_offset + 1];
                    const float nz = normal_camera[normal_offset + 2];
                    const float normal_norm = sqrtf(nx * nx + ny * ny + nz * nz);
                    const float normal_dot = normal_norm > 1.0e-6f
                        ? (ray_dx * nx + ray_dy * ny + ray_dz * nz) *
                          inv_ray_norm / normal_norm
                        : 0.0f;
                    const float allowed_jump = fmaxf(
                        max_depth_jump_absolute,
                        max_depth_jump_relative * fmaxf(scene_z, previous_scene_z)
                    );
                    // A near-surface restart follows an invalid-depth hole, so
                    // previous_scene_z is stale by definition. Its strict
                    // thickness test replaces the unavailable continuity test.
                    const bool depth_continuous = restarted_near_surface ||
                        fabsf(scene_z - previous_scene_z) <= allowed_jump;

                    // Normals are pre-oriented toward the primary camera.
                    // A reflection ray must therefore enter a visible/front
                    // face (R dot N < 0). A positive dot is a back-face-only
                    // observation, such as seeing a tabletop top from below.
                    if (normal_dot > backface_cos_threshold) {
                        if (!saw_backface) {
                            proxy_coords[2 * pixel_idx] = xi;
                            proxy_coords[2 * pixel_idx + 1] = yi;
                            hit_normal_dot[pixel_idx] = normal_dot;
                        }
                        saw_backface = true;
                    } else if (normal_dot < -frontface_cos_threshold &&
                               depth_continuous) {
                        hit_coords[2 * pixel_idx] = xi;
                        hit_coords[2 * pixel_idx + 1] = yi;
                        hit_status[pixel_idx] = 1;  // valid front-face hit
                        hit_normal_dot[pixel_idx] = normal_dot;
                        break;
                    } else {
                        saw_ambiguous = true;
                        if (!saw_backface)
                            hit_normal_dot[pixel_idx] = normal_dot;
                    }
                }

                previous_delta = delta;
                previous_scene_z = scene_z;
                has_previous = true;
            }

            if (hit_status[pixel_idx] != 1)
                hit_status[pixel_idx] = saw_backface ? 2 : (saw_ambiguous ? 3 : 0);
        });
    cudaDeviceSynchronize();
}
