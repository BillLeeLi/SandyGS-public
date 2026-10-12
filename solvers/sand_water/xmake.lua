set_project("sand-water-mixtures")
set_languages("cxx20")

add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", {outputdir = ".vscode"})

add_requires("cxxopts v3.1.1")
add_requires("eigen 3.4.0")
add_requires("libnpy")
add_requires("spdlog", {configs = {fmt_external = true}})
add_requires("tbb")
add_requires("yaml-cpp 0.7.0")

-- 所有模拟模块编译成同一个静态核心库。后续增加 water/sand/coupling 文件时，
-- demo、tests 和其他前端可共享实现，而不会互相依赖 main()。
target("sand_water_core")
    set_kind("static")
    add_headerfiles(
        "*.h",
        "simulation/*.h",
        "numerics/*.h",
        "physics/*.h",
        "scene/*.h",
        "diagnostics/*.h",
        "output/*.h")
    add_files(
        "Extrapolation.cpp",
        "simulation/*.cpp",
        "numerics/*.cpp",
        "physics/*.cpp",
        "scene/*.cpp",
        "diagnostics/*.cpp",
        "output/*.cpp")
    add_includedirs(".", {public = true})
    add_packages(
        "eigen",
        "libnpy",
        "spdlog",
        "tbb",
        "yaml-cpp",
        {public = true})

target("demo")
    -- 当前里程碑演示场景：用于人工检查导出协议、守恒量和分层可视化。
    set_kind("binary")
    add_deps("sand_water_core")
    add_files("main.cpp")
    add_packages("cxxopts")

target("tests")
    -- 不依赖 demo 的独立测试入口，便于持续集成只执行数值验收。
    set_kind("binary")
    add_deps("sand_water_core")
    add_files("tests/*.cpp")

target("benchmark")
    -- M7 无输出性能入口：只推进 Simulation，避免把 NPY/OBJ/渲染计入求解耗时。
    set_kind("binary")
    add_deps("sand_water_core")
    add_files("benchmarks/*.cpp")
    add_packages("cxxopts")
