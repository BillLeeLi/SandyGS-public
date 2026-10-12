#include <pybind11/pybind11.h>

#include <torch/extension.h>
#include "bvh.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("create_bvh", &create_bvh);
    m.def("trace_bvh", &trace_bvh);
    m.def("trace_bvh_opacity", &trace_bvh_opacity);
    m.def("accum_refl_ray_colors", &accum_refl_ray_colors);
    m.def("get_screen_refl_colors",&get_screen_refl_colors);
    m.def("trace_screen_reflections", &trace_screen_reflections);
}
