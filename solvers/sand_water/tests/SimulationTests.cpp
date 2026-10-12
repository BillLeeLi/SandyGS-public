#include "output/Output.h"
#include "simulation/Simulation.h"
#include "numerics/MacOperators.h"

#include <npy.hpp>

namespace
{
    using namespace Pivot;

    // M0 使用一个极小的无依赖测试框架，避免仅为了断言和分组引入大型库。
    // 任一 Require* 失败都会抛异常，由文件末尾的统一 runner 汇报。
    void Require(bool condition, std::string_view message)
    {
        if (!condition)
        {
            throw std::runtime_error(std::string(message));
        }
    }

    void RequireNear(double actual, double expected, double tolerance, std::string_view message)
    {
        if (std::abs(actual - expected) > tolerance)
        {
            throw std::runtime_error(fmt::format(
                "{}: expected {}, got {} (tolerance {})",
                message, expected, actual, tolerance));
        }
    }

    template <typename Function>
    void RequireThrows(Function &&function, std::string_view message)
    {
        try
        {
            function();
        }
        catch (std::exception const &)
        {
            return;
        }
        throw std::runtime_error(std::string(message));
    }

    void TestConfigValidation()
    {
        // 配置测试关注“尽早拒绝”会导致无内部域、无定义材料模型或显式发散的输入。
        SimulationConfig config;
        config.Validate();

        SimulationConfig badResolution = config;
        badResolution.grid.resolution = {2, 2};
        RequireThrows(
            [&] { badResolution.Validate(); },
            "resolution without an interior should be rejected");

        SimulationConfig badCohesion = config;
        badCohesion.physics.cohesion.phi2 = badCohesion.physics.cohesion.phi1;
        RequireThrows(
            [&] { badCohesion.Validate(); },
            "unordered saturation thresholds should be rejected");

        SimulationConfig badDiffusionStep = config;
        badDiffusionStep.physics.diffusionRate = 1.0;
        badDiffusionStep.time.waterDt = 1.0;
        RequireThrows(
            [&] { badDiffusionStep.Validate(); },
            "unstable fixed diffusion time step should be rejected");

        SimulationConfig badInternalFloor = config;
        badInternalFloor.time.internalForceHeightFloor = -1.0e-4;
        RequireThrows(
            [&] { badInternalFloor.Validate(); },
            "negative internal-force height floor should be rejected");

        SimulationConfig badInternalFraction = config;
        badInternalFraction.time.internalForceHeightFraction = -1.0e-4;
        RequireThrows(
            [&] { badInternalFraction.Validate(); },
            "negative internal-force height fraction should be rejected");

        SimulationConfig badSandContactTolerance = config;
        badSandContactTolerance.physics.sandTerrainContactTolerance = -1.0e-4;
        RequireThrows(
            [&] { badSandContactTolerance.Validate(); },
            "negative sand terrain contact tolerance should be rejected");

        SimulationConfig badExchangeScheme = config;
        badExchangeScheme.exchangeScheme =
            static_cast<SandWaterExchangeScheme>(99);
        RequireThrows(
            [&] { badExchangeScheme.Validate(); },
            "unknown sand-water exchange scheme should be rejected");

        SimulationConfig badSaturationMode = config;
        badSaturationMode.physics.saturationMode =
            static_cast<SaturationEvolutionMode>(99);
        RequireThrows(
            [&] { badSaturationMode.Validate(); },
            "unknown saturation evolution mode should be rejected");

        SimulationConfig unstableExplicitExchange = config;
        unstableExplicitExchange.physics.diffusionRate = 0.0;
        unstableExplicitExchange.physics.momentumExchangeRate = 1.0e8;
        unstableExplicitExchange.time.waterDt = 0.1;
        unstableExplicitExchange.time.sandSubsteps = 1;
        unstableExplicitExchange.time.useAdaptiveDt = false;
        RequireThrows(
            [&] { unstableExplicitExchange.Validate(); },
            "fixed explicit exchange step that overshoots should be rejected");
    }

    void TestDryDruckerPragerClosedFormProjection()
    {
        // 对无黏聚沙，当前应力残差回映射应与 Klár 2016 Eq. (27--28)
        // 的三分支闭式投影一致。这个测试用于排除“回映射近似”是干沙伪峰
        // 根因；含黏聚分支仍由一般的应力空间求根处理。
        PhysicalParameters parameters;
        parameters.cohesion.c0 = 0.0;
        parameters.cohesion.c1 = 0.0;
        parameters.cohesion.c2 = 0.0;
        SandConstitutive constitutive(parameters, 1.0e-10);

        double constexpr angle = 0.37;
        Matrix2d rotation;
        rotation << std::cos(angle), -std::sin(angle),
                    std::sin(angle), std::cos(angle);
        Vector2d const trialLogStrain(0.15, -0.25);
        Matrix2d const trial =
            rotation *
            trialLogStrain.array().exp().matrix().asDiagonal();

        Vector2d const deviator =
            trialLogStrain -
            0.5 * trialLogStrain.sum() * Vector2d::Ones();
        double const deltaGamma =
            deviator.norm() +
            (2.0 * parameters.lameLambda +
             2.0 * parameters.lameMu) /
                (2.0 * parameters.lameMu) *
            trialLogStrain.sum() *
            parameters.druckerPragerFriction;
        Vector2d const expectedLogStrain =
            trialLogStrain -
            deltaGamma * deviator / deviator.norm();
        Matrix2d const expected =
            rotation *
            expectedLogStrain.array().exp().matrix().asDiagonal();
        Matrix2d const projected = constitutive.Project(trial, 0.0);
        RequireNear(
            (projected - expected).norm(),
            0.0,
            2.0e-12,
            "dry Drucker-Prager side projection should match Klar");

        Matrix2d const tensileTrial =
            rotation *
            Vector2d(0.12, 0.02).array().exp().matrix().asDiagonal();
        RequireNear(
            (constitutive.Project(tensileTrial, 0.0) - rotation).norm(),
            0.0,
            2.0e-12,
            "dry tensile state should return to the cone tip");
    }

    void TestConstitutiveProjectionIdempotence()
    {
        // 已经回到屈服面内的弹性 F 再次投影时不应继续发生塑性流动。
        // M6.6 重复 1000 次，专门把“返回映射自身漂移”与时间步内的
        // 速度/摩擦算子分裂区分开来。
        PhysicalParameters parameters;
        parameters.cohesion.c0 = 0.0;
        parameters.cohesion.c1 = 0.0;
        parameters.cohesion.c2 = 0.0;
        SandConstitutive constitutive(parameters, 1.0e-10);

        double constexpr angle = 0.23;
        Matrix2d rotation;
        rotation << std::cos(angle), -std::sin(angle),
                    std::sin(angle), std::cos(angle);
        Matrix2d const trial =
            rotation *
            Vector2d(0.18, -0.27).array().exp().matrix().asDiagonal();
        bool firstProjectionChanged = false;
        Matrix2d const once =
            constitutive.Project(
                trial, 0.0, firstProjectionChanged);
        Require(
            firstProjectionChanged,
            "plastic trial state must report a changed projection");
        Matrix2d repeated = once;
        for (int iteration = 0; iteration < 1000; ++iteration)
        {
            repeated = constitutive.Project(repeated, 0.0);
        }

        RequireNear(
            (repeated - once).norm(),
            0.0,
            1.0e-11,
            "projected constitutive state must be idempotent");
        RequireNear(
            repeated.determinant(),
            once.determinant(),
            1.0e-12,
            "repeated projection must preserve det(F)");

        bool identityChanged = true;
        Matrix2d const identityProjection =
            constitutive.Project(
                Matrix2d::Identity(),
                0.0,
                identityChanged);
        Require(
            !identityChanged,
            "elastic identity must report an unchanged projection");
        RequireNear(
            (identityProjection - Matrix2d::Identity()).norm(),
            0.0,
            0.0,
            "unchanged projection must return the exact trial matrix");
    }

    void TestAdaptiveSandElasticCfl()
    {
        // 自适应步长必须考虑显式内力的弹性波，而不只考虑重力浅水波。
        SimulationConfig config;
        config.grid.resolution = {10, 10};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.time.waterDt = 0.1;
        config.time.sandSubsteps = 1;
        config.time.targetCfl = 0.5;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 0.2; },
            [](Vector2d const &) { return 0.0; });
        simulation.Step();

        double const waveSpeed = std::sqrt(
            (config.physics.lameLambda +
             2.0 * config.physics.lameMu) /
            config.physics.sandDensity);
        double const expectedLimit =
            config.time.targetCfl * config.grid.spacing / waveSpeed;
        Require(
            simulation.GetState().lastDt <= expectedLimit *
                (1.0 + 1.0e-12),
            "adaptive step must satisfy the sand elastic CFL limit");
    }

    void TestStateInitialization()
    {
        // 状态清零不仅指高度/速度为零，本构变量必须回到 F=P=I。
        StaggeredGrid grid(1, 0.1, Vector2i(8, 6));
        SimulationState state(grid);
        state.SetZero();

        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            RequireNear(state.terrain[cell], 0.0, 0.0, "terrain should initialize to zero");
            RequireNear(state.sand.height[cell], 0.0, 0.0, "sand should initialize to zero");
            RequireNear(
                state.sand.deformationGradient[cell].determinant(),
                1.0, 1.0e-14, "F should initialize to identity");
            RequireNear(
                (state.sand.stretch[cell] - Matrix2d::Identity()).norm(),
                0.0, 1.0e-14, "P should initialize to identity");
        });

        RequireNear(state.sand.velocity.GetMaxAbsComponent(), 0.0, 0.0, "sand velocity");
        RequireNear(state.pureWater.velocity.GetMaxAbsComponent(), 0.0, 0.0, "pure water velocity");
        RequireNear(state.mixedWater.velocity.GetMaxAbsComponent(), 0.0, 0.0, "mixed water velocity");
    }

    void TestBoundaryConditions()
    {
        // 人工线性标量场可以精确判断 Neumann 复制来源；常速度场用于区分
        // 自由滑移条件的法向归零和切向保持。
        StaggeredGrid grid(1, 0.1, Vector2i(8, 6));
        BoundaryConditions boundary(grid);
        GridData<double> height(grid.GetCellGrid());
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            height[cell] = 10.0 * cell.x() + cell.y();
        });
        boundary.ApplyHeightBoundary(height);

        RequireNear(height[Vector2i(0, 0)], 11.0, 0.0, "lower corner Neumann copy");
        RequireNear(height[Vector2i(7, 5)], 64.0, 0.0, "upper corner Neumann copy");
        RequireNear(height[Vector2i(3, 2)], 32.0, 0.0, "interior height must not change");

        SGridData<double> velocity(grid.GetFaceGrids(), Vector2d(2.0, 3.0));
        boundary.ApplyVelocityBoundary(velocity);

        ForEach(grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            if (boundary.IsNormalBoundaryFace(axis, face))
            {
                RequireNear(velocity[axis][face], 0.0, 0.0, "closed normal velocity");
            }
        });
        RequireNear(velocity[0][Vector2i(2, 0)], 2.0, 0.0, "free-slip x tangent copy");
        RequireNear(velocity[1][Vector2i(0, 2)], 3.0, 0.0, "free-slip y tangent copy");
    }

    void TestMasksAndDiagnostics()
    {
        // 单个活跃单元使预期体积能够手算，并同时覆盖三种材料掩码。
        SimulationConfig config;
        config.grid.resolution = {8, 6};
        config.grid.spacing = 0.1;
        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        Diagnostics diagnostics;

        Vector2i const activeCell(2, 2);
        state.sand.height[activeCell] = 0.2;
        state.mixedWater.height[activeCell] = 0.1;
        state.pureWater.height[activeCell] = 0.05;
        boundary.ApplyVelocityBoundary(state.sand.velocity);
        boundary.ApplyVelocityBoundary(state.pureWater.velocity);
        boundary.ApplyVelocityBoundary(state.mixedWater.velocity);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);

        Require(scratch.sandActive[activeCell] == 1, "sand mask should mark occupied cell");
        Require(scratch.mixedWaterActive[activeCell] == 1, "mixed mask should mark occupied cell");
        Require(scratch.pureWaterActive[activeCell] == 1, "pure mask should mark occupied cell");
        Require(scratch.faceValid[0][activeCell] == 1, "adjacent sand face should be valid");

        diagnostics.Validate(state, config, grid, boundary, "diagnostics test");
        ConservationReport const report = diagnostics.Measure(state, config, grid);
        RequireNear(report.sandVolume, 0.002, 1.0e-14, "sand volume");
        RequireNear(report.mixedWaterVolume, 0.001, 1.0e-14, "mixed volume");
        RequireNear(report.pureWaterVolume, 0.0005, 1.0e-14, "pure volume");
        RequireNear(report.minDetF, 1.0, 1.0e-14, "minimum determinant");

        state.mixedWater.height[activeCell] = 0.3;
        RequireThrows(
            [&] { diagnostics.Validate(state, config, grid, boundary, "invalid mixed height"); },
            "mixed water above sand height should be rejected");
    }

    void TestSimulationInitialization()
    {
        // 总水 0.40 在沙高 0.25 处应拆成 mixed=0.25、pure=0.15。
        SimulationConfig config;
        config.grid.resolution = {8, 6};
        config.grid.spacing = 0.1;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.1; },
            [](Vector2d const &) { return 0.25; },
            [](Vector2d const &) { return 0.40; });

        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            RequireNear(simulation.GetState().mixedWater.height[cell], 0.25, 1.0e-14,
                        "mixed water split");
            RequireNear(simulation.GetState().pureWater.height[cell], 0.15, 1.0e-14,
                        "pure water split");
        });
        simulation.Validate("uniform initialized simulation");
    }

    void TestNpyOutput()
    {
        // 集成检查覆盖 Simulation -> Diagnostics -> FrameExporter 的完整 M0 链路，
        // 并用独立 NPY 读取函数验证外部工具实际能够读取数组。
        SimulationConfig config;
        config.grid.resolution = {8, 6};
        config.grid.spacing = 0.1;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &position) { return position.x(); },
            [](Vector2d const &position) { return position.squaredNorm() < 0.04 ? 0.2 : 0.0; },
            [](Vector2d const &position) { return position.x() < 0.0 ? 0.1 : 0.0; });

        auto const unique = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path const runDirectory =
            std::filesystem::temp_directory_path() /
            fmt::format("sand-water-m0-test-{}", unique);

        FrameExporter exporter;
        exporter.WriteMetadata(
            config,
            runDirectory,
            "output_test",
            SnapshotLevel::Diagnostic);
        std::filesystem::path const frameDirectory = exporter.WriteFrame(
            SimulationSnapshot(
                simulation.GetState(),
                simulation.GetScratch()),
            simulation.Measure(),
            config,
            SnapshotLevel::Diagnostic,
            runDirectory);
        std::filesystem::path const meshDirectory =
            runDirectory / "meshes" / "000000";
        exporter.WriteObjMeshes(
            SimulationSnapshot(simulation.GetState()),
            config,
            meshDirectory);

        Require(std::filesystem::exists(runDirectory / "metadata.yaml"), "metadata missing");
        Require(std::filesystem::exists(runDirectory / "metrics.csv"), "metrics missing");
        Require(std::filesystem::exists(frameDirectory / "sand_height.npy"), "sand NPY missing");
        Require(std::filesystem::exists(frameDirectory / "det_f.npy"), "det(F) NPY missing");
        Require(std::filesystem::exists(frameDirectory / "stress.npy"), "stress NPY missing");
        Require(
            std::filesystem::exists(frameDirectory / "yield_residual.npy"),
            "yield residual NPY missing");
        Require(
            std::filesystem::exists(
                frameDirectory / "internal_yield_residual.npy"),
            "internal-stage yield residual NPY missing");
        Require(
            std::filesystem::exists(
                frameDirectory / "internal_normal_acceleration_velocity_x.npy"),
            "internal normal acceleration NPY missing");
        Require(
            std::filesystem::exists(
                frameDirectory / "internal_tangential_acceleration_velocity_y.npy"),
            "internal tangential acceleration NPY missing");
        Require(
            std::filesystem::exists(
                frameDirectory / "internal_effective_height_velocity_x.npy"),
            "internal effective-height NPY missing");
        Require(
            std::filesystem::exists(meshDirectory / "terrain.obj"),
            "terrain OBJ missing");
        Require(
            std::filesystem::exists(meshDirectory / "sand_surface.obj"),
            "sand OBJ missing");
        Require(
            std::filesystem::exists(meshDirectory / "water_surface.obj"),
            "water OBJ missing");

        std::vector<unsigned long> shape;
        std::vector<double> data;
        npy::LoadArrayFromNumpy<double>(
            (frameDirectory / "sand_height.npy").string(), shape, data);
        Require(shape == std::vector<unsigned long>({8, 6}), "NPY shape mismatch");
        Require(data.size() == 48, "NPY element count mismatch");

        // M6.5 Render 快照应保留三维曲面所需高度，但不写六张速度和本构
        // 大数组。这项检查防止后续修改意外让长程输出重新膨胀。
        std::filesystem::path const renderDirectory =
            runDirectory / "render_only";
        exporter.WriteMetadata(
            config,
            renderDirectory,
            "render_output_test",
            SnapshotLevel::Render);
        std::filesystem::path const renderFrame =
            exporter.WriteFrame(
                SimulationSnapshot(simulation.GetState()),
                simulation.Measure(),
                config,
                SnapshotLevel::Render,
                renderDirectory);
        Require(
            std::filesystem::exists(
                renderFrame / "total_water.npy"),
            "render snapshot total-water height missing");
        Require(
            !std::filesystem::exists(
                renderFrame / "sand_velocity_x.npy"),
            "render snapshot must omit MAC velocities");
        Require(
            !std::filesystem::exists(
                renderFrame / "deformation_gradient.npy"),
            "render snapshot must omit constitutive arrays");

        // WriteMetrics 必须能脱离 WriteFrame 单独追加，长程主循环正是用该接口
        // 保留每一步标量历史而只按 stride 写大型数组。
        exporter.WriteMetrics(
            SimulationSnapshot(simulation.GetState()),
            simulation.Measure(),
            renderDirectory);
        std::ifstream metrics(renderDirectory / "metrics.csv");
        std::size_t metricLineCount = 0;
        std::string metricLine;
        while (std::getline(metrics, metricLine))
        {
            ++metricLineCount;
        }
        Require(
            metricLineCount == 3,
            "metrics should contain header, frame row, and explicit row");

        std::filesystem::remove_all(runDirectory);
    }

    void TestM65MaximumStep()
    {
        // duration 模式依靠公开的 maximumDt 缩短最后一步。关闭所有动力学后，
        // 时间标签和 lastDt 应精确等于调用者给出的上限。
        SimulationConfig config;
        config.grid.resolution = {8, 6};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.enableSandInternalForce = false;
        config.time.waterDt = 0.01;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);

        simulation.Step(0.0037);
        RequireNear(
            simulation.GetState().time,
            0.0037, 1.0e-15, "M6.5 duration time");
        RequireNear(
            simulation.GetState().lastDt,
            0.0037, 1.0e-15, "M6.5 shortened dt");
        Require(
            simulation.GetState().stepIndex == 1,
            "M6.5 maximum step should advance exactly once");
    }

    void TestSceneSources()
    {
        // 三种 M6 Source 共用一个中心 box。先生成沙，再注水，最后对三相
        // 规定速度，从而分别检查高度增量、mixed 容量划分和材料选择逻辑。
        SimulationConfig config;
        config.grid.resolution = {10, 8};
        config.grid.spacing = 0.1;
        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SourceRegion region;
        region.shape = SourceRegion::Shape::Box;
        region.minimum = {-0.2, -0.2};
        region.maximum = {0.2, 0.2};

        SandEmitter const sandEmitter(
            region, 2.0, Vector2d(0.3, 0.0), 0.0, 1.0);
        WaterInlet const waterInlet(
            region, 1.0, Vector2d(0.5, 0.0), 0.0, 1.0);
        VelocityRegion const velocityRegion(
            region,
            SourceMaterial::All,
            Vector2d(0.7, -0.2),
            0.0,
            1.0);
        sandEmitter.Apply(state, 0.0, 0.1);
        waterInlet.Apply(state, 0.0, 0.1);
        velocityRegion.Apply(state, 0.0, 0.1);

        Vector2i const centerCell(5, 4);
        RequireNear(
            state.sand.height[centerCell],
            0.2, 1.0e-14, "sand emitter height");
        RequireNear(
            state.mixedWater.height[centerCell],
            0.1, 1.0e-14, "water inlet mixed fill");
        RequireNear(
            state.pureWater.height[centerCell],
            0.0, 1.0e-14, "water inlet pure remainder");
        RequireNear(
            state.sand.velocity[0][Vector2i(5, 4)],
            0.7, 1.0e-14, "velocity region sand x");
        RequireNear(
            state.mixedWater.velocity[1][Vector2i(5, 4)],
            -0.2, 1.0e-14, "velocity region mixed-water y");
    }

    void TestSceneLoaderAndScheduling()
    {
        // 临时 YAML 同时覆盖配置、解析几何和 WaterInlet。随后交给 Simulation
        // 验证源项确实在水步之前按重叠时长增加总水量。
        auto const unique =
            std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path const directory =
            std::filesystem::temp_directory_path() /
            fmt::format("sand-water-m6-scene-test-{}", unique);
        std::filesystem::create_directories(directory);
        std::filesystem::path const yamlPath =
            directory / "scene.yaml";
        {
            std::ofstream yaml(yamlPath);
            yaml <<
                "name: loader_test\n"
                "grid:\n"
                "  resolution: [12, 10]\n"
                "  spacing: 0.1\n"
                "physics:\n"
                "  gravity: 0.0\n"
                "  diffusion_rate: 0.0\n"
                "  momentum_exchange_rate: 0.0\n"
                "  ground_friction: 0.0\n"
                "  sand_terrain_contact_tolerance: 0.005\n"
                "  enable_sand_internal_force: false\n"
                "  project_deformation_after_advection: true\n"
                "  constitutive_saturation_lowpass: true\n"
                "time:\n"
                "  water_dt: 0.01\n"
                "  sand_substeps: 1\n"
                "  adaptive: true\n"
                "initial:\n"
                "  terrain: {type: plane, height: 0.05, slope: [0.0, 0.0]}\n"
                "  sand: {type: box, min: [-0.2, -0.2], max: [0.2, 0.2], height: 0.2, edge_width: 0.2}\n"
                "  total_water: 0.0\n"
                "sources:\n"
                "  - type: water_inlet\n"
                "    region: {type: circle, center: [0.0, 0.0], radius: 0.18}\n"
                "    height_rate: 1.0\n"
                "    velocity: [0.0, 0.0]\n"
                "    begin: 0.0\n"
                "    end: 0.1\n"
                "run: {steps: 2, duration: 0.025, frame_stride: 1, write_obj: true}\n";
        }

        SceneDescription scene =
            SceneLoader::FromYaml(yamlPath);
        Require(
            scene.run.name == "loader_test" &&
                scene.run.steps == 2 &&
                scene.run.writeObj,
            "scene run settings should load from YAML");
        RequireNear(
            scene.run.duration,
            0.025, 1.0e-14, "scene duration");
        RequireNear(
            scene.terrain(Vector2d::Zero()),
            0.05, 1.0e-14, "scene plane height");
        RequireNear(
            scene.sandHeight(Vector2d::Zero()),
            0.2, 1.0e-14, "scene box height");
        RequireNear(
            scene.sandHeight(Vector2d(-0.2, 0.0)),
            0.1, 1.0e-14, "smoothed box boundary half coverage");
        RequireNear(
            scene.sandHeight(Vector2d(-0.31, 0.0)),
            0.0, 1.0e-14, "smoothed box outside transition");
        RequireNear(
            scene.config.physics.sandTerrainContactTolerance,
            0.005, 1.0e-14, "sand terrain contact tolerance");
        Require(
            scene.config.physics.saturationMode ==
                SaturationEvolutionMode::Dynamic,
            "scene loader should preserve dynamic saturation mode by default");
        Require(
            scene.config.physics.projectDeformationAfterAdvection,
            "scene loader should parse post-advection constitutive projection");
        Require(
            scene.config.physics.constitutiveSaturationLowpass,
            "scene loader should parse constitutive saturation lowpass");

        SimulationConfig const loadedConfig = scene.config;
        Simulation simulation(loadedConfig);
        simulation.Initialize(std::move(scene));
        double const waterBefore =
            simulation.Measure().totalWaterVolume;
        simulation.Step();
        Require(
            simulation.Measure().totalWaterVolume >
                waterBefore,
            "scheduled water inlet must add volume before the water solve");

        // NPY primitive 使用相对 YAML 路径并进行双线性读取。
        std::vector<unsigned long> const mapShape{4, 4};
        std::vector<double> const mapValues(16, 0.37);
        npy::SaveArrayAsNumpy(
            (directory / "height.npy").string(),
            false,
            2,
            mapShape.data(),
            mapValues);
        std::filesystem::path const npyYamlPath =
            directory / "npy_scene.yaml";
        {
            std::ofstream yaml(npyYamlPath);
            yaml <<
                "grid: {resolution: [4, 4], spacing: 0.1}\n"
                "physics: {diffusion_rate: 0.0, saturation_mode: frozen_bound}\n"
                "initial:\n"
                "  sand: {type: npy, file: height.npy}\n";
        }
        SceneDescription npyScene =
            SceneLoader::FromYaml(npyYamlPath);
        RequireNear(
            npyScene.sandHeight(Vector2d::Zero()),
            0.37, 1.0e-14, "NPY scene height");
        Require(
            npyScene.config.physics.saturationMode ==
                SaturationEvolutionMode::FrozenBound,
            "scene loader should parse frozen_bound saturation mode");

        std::filesystem::remove_all(directory);
    }

    double InteriorVolume(
        GridData<double> const &height,
        StaggeredGrid const &grid)
    {
        double volume = 0.0;
        double const area = grid.GetSpacing() * grid.GetSpacing();
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (grid.IsInsideCell(cell))
            {
                volume += height[cell] * area;
            }
        });
        return volume;
    }

    void TestMacOperators()
    {
        // 线性场 f=2x-3y+1 的面中心梯度应被一阶差分精确恢复。
        StaggeredGrid grid(1, 0.1, Vector2i(10, 8));
        GridData<double> field(grid.GetCellGrid());
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            Vector2d const position = grid.GetCellGrid().PositionOf(cell);
            field[cell] = 2.0 * position.x() - 3.0 * position.y() + 1.0;
        });

        RequireNear(
            MacOperators::GradientAtFace(field, 0, Vector2i(4, 3)),
            2.0, 1.0e-13, "linear x gradient at MAC face");
        RequireNear(
            MacOperators::GradientAtFace(field, 1, Vector2i(4, 3)),
            -3.0, 1.0e-13, "linear y gradient at MAC face");
    }

    void TestFiniteVolumeHeightAdvection()
    {
        // 单个高度脉冲在 CFL=0.1 的均匀 x 流中，一阶迎风更新有精确期望值：
        // donor 保留 0.9，右邻单元得到 0.1。
        StaggeredGrid grid(1, 0.1, Vector2i(12, 8));
        BoundaryConditions boundary(grid);
        GridData<double> source(grid.GetCellGrid());
        GridData<double> destination(grid.GetCellGrid());
        SGridData<double> velocity(grid.GetFaceGrids(), Vector2d(0.2, 0.0));
        Vector2i const donor(4, 3);
        source[donor] = 1.0;
        boundary.ApplyHeightBoundary(source);
        boundary.ApplyVelocityBoundary(velocity);

        double const volumeBefore = InteriorVolume(source, grid);
        AdvectionStats const stats = AdvectionFV::SolveHeight(
            source, velocity, 0.05, boundary, destination);
        double const volumeAfter = InteriorVolume(destination, grid);

        RequireNear(destination[donor], 0.9, 1.0e-14, "upwind donor value");
        RequireNear(
            destination[donor + Vector2i::UnitX()],
            0.1, 1.0e-14, "upwind receiver value");
        RequireNear(volumeAfter, volumeBefore, 1.0e-14, "closed FV mass");
        RequireNear(stats.VolumeChange(), 0.0, 1.0e-14, "reported FV mass");
        Require(stats.minValue >= 0.0, "FV height must remain non-negative");

        GridData<Matrix2d> matrixSource(
            grid.GetCellGrid(), Matrix2d::Identity());
        GridData<Matrix2d> matrixDestination(grid.GetCellGrid());
        SGridData<double> zeroVelocity(grid.GetFaceGrids());
        AdvectionFV::SolveConserved(
            matrixSource,
            zeroVelocity,
            0.05,
            boundary,
            matrixDestination);
        RequireNear(
            (matrixDestination[donor] - Matrix2d::Identity()).norm(),
            0.0,
            1.0e-14,
            "generic matrix conservative advection");

        // 极大 CFL 会触发 donor 流出限制器；质量仍守恒且 donor 不会变负。
        AdvectionFV::SolveHeight(
            source, velocity, 2.0, boundary, destination);
        Require(
            destination[donor] >= 0.0,
            "outflow limiter must prevent negative donor height");
        RequireNear(
            InteriorVolume(destination, grid),
            volumeBefore, 1.0e-14, "limited FV mass");
    }

    void TestSemiLagrangianZeroFlow()
    {
        // 零回溯流场的 departure point 就是当前面位置，任意源速度应保持不变。
        StaggeredGrid grid(1, 0.1, Vector2i(10, 8));
        BoundaryConditions boundary(grid);
        SGridData<double> source(grid.GetFaceGrids());
        SGridData<double> zeroFlow(grid.GetFaceGrids());
        SGridData<double> destination(grid.GetFaceGrids());
        ForEach(grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            Vector2d const position = grid.GetFaceGrids()[axis].PositionOf(face);
            source[axis][face] =
                axis == 0 ? 0.3 * position.x() : -0.2 * position.y();
        });

        AdvectionSL::Solve(source, zeroFlow, 0.25, boundary, destination);
        ForEach(grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            if (boundary.IsInsideFace(axis, face))
            {
                RequireNear(
                    destination[axis][face],
                    source[axis][face],
                    1.0e-14,
                    "zero-flow semi-Lagrangian identity");
            }
        });
    }

    void TestMixedWaterDiffusion()
    {
        // alpha=D*dt/dx^2=0.1。二维五点模板作用于单位脉冲后，中心应为
        // 1-4alpha=0.6，四个轴向邻居各为 0.1；Neumann 封闭域总质量不变。
        SimulationConfig config;
        config.grid.resolution = {11, 11};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.01;
        config.time.waterDt = 0.1;
        config.time.useAdaptiveDt = false;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 1.0; },
            [](Vector2d const &position)
            {
                return position.squaredNorm() < 1.0e-6 ? 1.0 : 0.0;
            });

        double const initialVolume = simulation.Measure().mixedWaterVolume;
        simulation.Step();

        Vector2i const center(5, 5);
        SimulationState const &state = simulation.GetState();
        RequireNear(
            state.mixedWater.height[center],
            0.6, 1.0e-14, "diffusion pulse center");
        for (int ordinal = 0; ordinal < Grid::GetNumNeighbors(); ++ordinal)
        {
            RequireNear(
                state.mixedWater.height[Grid::NeighborOf(center, ordinal)],
                0.1, 1.0e-14, "diffusion pulse axial neighbor");
        }
        RequireNear(
            simulation.Measure().mixedWaterVolume,
            initialVolume,
            1.0e-14,
            "Neumann diffusion mass");
        RequireNear(
            simulation.GetLastWaterStepReport().DiffusionVolumeChange(),
            0.0,
            1.0e-14,
            "reported diffusion mass");
        RequireNear(
            state.mixedWater.velocity.GetMaxAbsComponent(),
            0.0,
            0.0,
            "zero velocity must remain zero through diffusion");

        // 常量场的 Laplacian 为零，是独立于脉冲模板的零空间检查。
        Simulation constantSimulation(config);
        constantSimulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 1.0; },
            [](Vector2d const &) { return 0.2; });
        constantSimulation.Step();
        ForEach(constantSimulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (constantSimulation.GetGrid().IsInsideCell(cell))
            {
                RequireNear(
                    constantSimulation.GetState().mixedWater.height[cell],
                    0.2, 1.0e-14, "constant diffusion field");
            }
        });
    }

    void TestWaterLayerTransfer()
    {
        // 先增大沙容量触发 pure -> mixed，再减小容量触发 mixed -> pure。
        // 两次都使用容易手算的速度，检查接收层按 donor 动量重组且总动量不变。
        SimulationConfig config;
        config.grid.resolution = {8, 6};
        config.grid.spacing = 0.1;
        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        WaterSolver solver;

        state.sand.height.SetConstant(0.2);
        state.mixedWater.height.SetConstant(0.2);
        state.pureWater.height.SetConstant(0.3);
        state.pureWater.velocity.SetConstant(Vector2d(2.0, 0.0));
        state.mixedWater.velocity.SetConstant(Vector2d(-1.0, 0.0));
        boundary.ApplyVelocityBoundary(state.pureWater.velocity);
        boundary.ApplyVelocityBoundary(state.mixedWater.velocity);

        state.sand.height.SetConstant(0.3);
        WaterStepReport const filling = solver.CorrectLayersAfterSandMotion(
            state, config, 0.01, boundary, scratch);
        Vector2i const sampleFace(3, 3);
        RequireNear(
            state.pureWater.height[Vector2i(3, 3)],
            0.2, 1.0e-14, "pure height after filling pores");
        RequireNear(
            state.mixedWater.height[Vector2i(3, 3)],
            0.3, 1.0e-14, "mixed height after filling pores");
        RequireNear(
            state.pureWater.velocity[0][sampleFace],
            2.0, 1.0e-14, "pure donor velocity");
        RequireNear(
            state.mixedWater.velocity[0][sampleFace],
            0.0, 1.0e-14, "mixed receiver momentum");
        RequireNear(
            filling.totalVolumeAfterLayerCorrection,
            filling.totalVolumeBefore,
            1.0e-14, "layer correction total volume");
        RequireNear(
            filling.maxAbsLayerTransfer,
            0.1, 1.0e-14, "reported pure-to-mixed transfer");
        Require(
            filling.maxLayerMomentumError < 1.0e-12,
            "pure-to-mixed momentum must be conserved");

        state.sand.height.SetConstant(0.1);
        WaterStepReport const draining = solver.CorrectLayersAfterSandMotion(
            state, config, 0.01, boundary, scratch);
        RequireNear(
            state.pureWater.height[Vector2i(3, 3)],
            0.4, 1.0e-14, "pure height after draining pores");
        RequireNear(
            state.mixedWater.height[Vector2i(3, 3)],
            0.1, 1.0e-14, "mixed height after draining pores");
        RequireNear(
            state.pureWater.velocity[0][sampleFace],
            1.0, 1.0e-14, "pure receiver momentum");
        RequireNear(
            state.mixedWater.velocity[0][sampleFace],
            0.0, 1.0e-14, "mixed donor velocity");
        RequireNear(
            draining.maxAbsLayerTransfer,
            0.2, 1.0e-14, "reported mixed-to-pure transfer");
        Require(
            draining.maxLayerMomentumError < 1.0e-12,
            "mixed-to-pure momentum must be conserved");
    }

    void TestLakeAtRest()
    {
        // H+h 为常数时静水压力梯度严格为零；这是浅水实现最重要的平衡态测试。
        SimulationConfig config;
        config.grid.resolution = {24, 16};
        config.grid.spacing = 0.05;
        config.time.waterDt = 0.01;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &position)
            {
                return 0.04 * position.x();
            },
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return 0.35 - 0.04 * position.x();
            });

        double const initialVolume = simulation.Measure().totalWaterVolume;
        for (int step = 0; step < 25; ++step)
        {
            simulation.Step();
        }

        ConservationReport const report = simulation.Measure();
        Require(
            report.maxVelocity < 1.0e-11,
            "lake-at-rest should not create spurious velocity");
        RequireNear(
            report.totalWaterVolume,
            initialVolume,
            1.0e-12,
            "lake-at-rest water volume");
        Require(report.minHeight >= -1.0e-14, "lake-at-rest height");

        double const targetTime = simulation.GetState().time + 0.0037;
        simulation.AdvanceTo(targetTime);
        RequireNear(
            simulation.GetState().time,
            targetTime,
            0.0,
            "AdvanceTo exact target time");
    }

    void TestLakeAtRestAroundDryElevatedIsland()
    {
        // 湿/干边界的静水平衡回归：中央高地高于自由水面，因而应当作为
        // 不可穿透的干燥障碍物，而不能被解释成一个更高的水力势液柱。
        // 该测试刻意只推进一个时间步，以便把首个伪速度定位到水压更新，
        // 避免后续平流和网格色散掩盖问题来源。
        SimulationConfig config;
        config.grid.resolution = {32, 24};
        config.grid.spacing = 0.05;
        config.time.waterDt = 0.005;
        config.time.useAdaptiveDt = false;

        double constexpr waterSurface = 0.30;
        double constexpr islandTerrain = 0.40;
        auto const isIsland = [](Vector2d const &position)
        {
            return std::abs(position.x()) <= 0.10 &&
                   std::abs(position.y()) <= 0.25;
        };

        Simulation simulation(config);
        simulation.InitializeHeights(
            [&](Vector2d const &position)
            {
                return isIsland(position) ? islandTerrain : 0.0;
            },
            [](Vector2d const &) { return 0.0; },
            [&](Vector2d const &position)
            {
                // 低地具有恒定绝对水位；高地保持完全干燥。
                return isIsland(position) ? 0.0 : waterSurface;
            });

        double const initialVolume = simulation.Measure().totalWaterVolume;
        simulation.Step();

        ConservationReport const report = simulation.Measure();
        RequireNear(
            report.maxVelocity,
            0.0,
            1.0e-11,
            "dry elevated island must not repel a lake at rest");
        RequireNear(
            report.totalWaterVolume,
            initialVolume,
            1.0e-12,
            "dry-island lake-at-rest water volume");
        Require(
            report.minHeight >= -1.0e-14,
            "dry-island lake-at-rest height must remain non-negative");
    }

    void TestTwoLayerPressureAcrossPureMixedPartition()
    {
        // 这个测试只运行 WaterSolver：右半域的沙只提供固定孔隙容量，不参与
        // 沙动力学。两侧总水位存在已知小梯度，但水相分别是 pure 和 mixed；
        // 两层必须读取同一个总水压力势，不能把另一层为零误判成固体高地。
        SimulationConfig config;
        config.grid.resolution = {12, 8};
        config.grid.spacing = 0.1;
        config.physics.gravity = 10.0;
        config.physics.diffusionRate = 0.0;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        WaterSolver solver;
        state.SetZero();

        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            double const x = grid.GetCellGrid().PositionOf(cell).x();
            if (x < 0.0)
            {
                state.pureWater.height[cell] = 0.30;
            }
            else
            {
                state.sand.height[cell] = 0.31;
                state.mixedWater.height[cell] = 0.31;
            }
        });

        double constexpr dt = 0.001;
        solver.Step(state, config, dt, boundary, scratch);

        // 偶数分辨率下 x=0 是第 resolution.x()/2 个 x-face。
        Vector2i const interfaceFace(
            config.grid.resolution.x() / 2,
            config.grid.resolution.y() / 2);
        double const expectedVelocity =
            -config.physics.gravity * dt * (0.31 - 0.30) /
            config.grid.spacing;
        RequireNear(
            state.pureWater.velocity[0][interfaceFace],
            expectedVelocity,
            1.0e-13,
            "pure layer must feel shared pressure across a mixed-only neighbor");
        RequireNear(
            state.mixedWater.velocity[0][interfaceFace],
            expectedVelocity,
            1.0e-13,
            "mixed layer must feel shared pressure across a pure-only neighbor");
    }

    void TestMixedWaterAtRestAroundDryElevatedIsland()
    {
        // 与 pure-water 高地回归相同，但周围水完全位于固定沙孔隙中。
        // 直接运行 WaterSolver 可排除沙重力、本构、摩擦和动量交换的影响，
        // 从而验证两层水是否共享同一个“总水深 + 地形”接触条件。
        SimulationConfig config;
        config.grid.resolution = {32, 24};
        config.grid.spacing = 0.05;
        config.physics.diffusionRate = 0.0;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        WaterSolver solver;
        state.SetZero();

        double constexpr waterSurface = 0.30;
        double constexpr islandTerrain = 0.40;
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            Vector2d const position = grid.GetCellGrid().PositionOf(cell);
            bool const isIsland =
                std::abs(position.x()) <= 0.10 &&
                std::abs(position.y()) <= 0.25;
            if (isIsland)
            {
                state.terrain[cell] = islandTerrain;
            }
            else
            {
                // 沙容量与水深相等，确保层修正是恒等映射且不存在 pure water。
                state.sand.height[cell] = waterSurface;
                state.mixedWater.height[cell] = waterSurface;
            }
        });

        solver.Step(state, config, 0.005, boundary, scratch);
        RequireNear(
            state.mixedWater.velocity.GetMaxAbsComponent(),
            0.0,
            1.0e-11,
            "dry elevated island must not repel mixed water at rest");
        RequireNear(
            state.pureWater.height.GetMaxAbsValue(),
            0.0,
            0.0,
            "mixed-water obstacle test must not create a pure-water layer");
    }

    void TestWaterCannotAdvectIntoDryElevatedTerrain()
    {
        // 关闭重力并人为留下朝高地的旧速度，专门验证接触条件发生在高度平流
        // 之前；若只在压力阶段跳过梯度，右侧高地会先收到错误水通量。
        SimulationConfig config;
        config.grid.resolution = {12, 8};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        WaterSolver solver;
        state.SetZero();
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (grid.GetCellGrid().PositionOf(cell).x() < 0.0)
            {
                state.pureWater.height[cell] = 0.30;
            }
            else
            {
                state.terrain[cell] = 0.40;
            }
        });
        state.pureWater.velocity.SetConstant(Vector2d(0.20, 0.0));
        boundary.ApplyVelocityBoundary(state.pureWater.velocity);

        solver.Step(state, config, 0.01, boundary, scratch);
        Vector2i const firstHighCell(
            config.grid.resolution.x() / 2,
            config.grid.resolution.y() / 2);
        Vector2i const interfaceFace = firstHighCell;
        RequireNear(
            state.pureWater.height[firstHighCell],
            0.0,
            0.0,
            "water contact must block flux before height advection");
        RequireNear(
            state.pureWater.velocity[0][interfaceFace],
            0.0,
            0.0,
            "water contact must clear stale normal velocity at a high wall");
    }

    void TestDamBreakShortRun()
    {
        // 小网格短程集成同时覆盖 wet/dry 前沿、压力加速、反复平流和逐步诊断。
        SimulationConfig config;
        config.grid.resolution = {36, 18};
        config.grid.spacing = 0.05;
        config.time.waterDt = 0.006;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return position.x() < -0.15 ? 0.30 : 0.0;
            });

        double const initialVolume = simulation.Measure().totalWaterVolume;
        double initialMoment = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell))
            {
                initialMoment +=
                    simulation.GetState().pureWater.height[cell] *
                    simulation.GetGrid().GetCellGrid().PositionOf(cell).x();
            }
        });

        for (int step = 0; step < 80; ++step)
        {
            simulation.Step();
        }

        ConservationReport const report = simulation.Measure();
        double finalMoment = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell))
            {
                finalMoment +=
                    simulation.GetState().pureWater.height[cell] *
                    simulation.GetGrid().GetCellGrid().PositionOf(cell).x();
            }
        });

        Require(
            finalMoment > initialMoment,
            "dam-break water center should move toward the dry side");
        Require(report.maxVelocity > 0.0, "dam-break must generate water velocity");
        Require(report.minHeight >= -1.0e-14, "dam-break height must stay non-negative");
        RequireNear(
            report.totalWaterVolume,
            initialVolume,
            2.0e-12,
            "dam-break closed-domain water volume");
        simulation.Validate("M1 short dam-break");
    }

    void TestTwoLayerWaterShortRun()
    {
        // 静态沙床中的左侧水库同时含 pure/mixed 两层。短程运行应保持总水质量
        // 与容量约束，并因扩散、不同占据区和 donor 重组产生可测的层速度差。
        SimulationConfig config;
        config.grid.resolution = {40, 20};
        config.grid.spacing = 0.05;
        // M2 回归在 M3 接入后仍需隔离沙动力学，因此关闭重力并人为给 pure 层
        // 初速；沙摩擦也随 g=0 关闭。
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 2.0e-3;
        config.time.waterDt = 0.006;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 0.22; },
            [](Vector2d const &position)
            {
                // 域平均总水高大于沙容量，长时间重新分布后仍会保留 pure 层。
                return position.x() < -0.2 ? 0.48 : 0.08;
            });
        ForEach(simulation.GetGrid().GetFaceGrids(),
            [&](int axis, Vector2i const &face)
        {
            if (axis == 0 &&
                !simulation.GetGrid().IsBoundaryFace(axis, face))
            {
                simulation.GetState().pureWater.velocity[axis][face] = 0.25;
            }
        });

        double const initialWaterVolume =
            simulation.Measure().totalWaterVolume;
        for (int step = 0; step < 100; ++step)
        {
            simulation.Step();
        }

        ConservationReport const report = simulation.Measure();
        double maxLayerVelocityDifference = 0.0;
        ForEach(simulation.GetGrid().GetFaceGrids(),
            [&](int axis, Vector2i const &face)
        {
            if (simulation.GetGrid().IsInsideFace(axis, face))
            {
                maxLayerVelocityDifference = std::max(
                    maxLayerVelocityDifference,
                    std::abs(
                        simulation.GetState().pureWater.velocity[axis][face] -
                        simulation.GetState().mixedWater.velocity[axis][face]));
            }
        });

        RequireNear(
            report.totalWaterVolume,
            initialWaterVolume,
            3.0e-12,
            "M2 closed-domain total water volume");
        Require(
            report.pureWaterVolume > 0.0 && report.mixedWaterVolume > 0.0,
            "M2 run must retain both water layers");
        Require(
            maxLayerVelocityDifference > 1.0e-4,
            "pure and mixed water velocities should visibly separate");
        Require(
            report.minHeight >= -1.0e-14,
            "M2 water heights must remain non-negative");
        simulation.Validate("M2 short two-layer water run");
    }

    void TestSandPotentialAtRest()
    {
        // H=0.04x、hs=0.30-0.04x 令 H+hs 精确为常数。无水情况下沙重力势
        // 梯度应严格为零，因此速度、高度和质量都不能产生伪变化。
        SimulationConfig config;
        config.grid.resolution = {24, 16};
        config.grid.spacing = 0.05;
        config.physics.diffusionRate = 0.0;
        config.physics.groundFriction = 0.5;
        config.time.waterDt = 0.006;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &position)
            {
                return 0.04 * position.x();
            },
            [](Vector2d const &position)
            {
                return 0.30 - 0.04 * position.x();
            },
            [](Vector2d const &) { return 0.0; });

        double const initialVolume = simulation.Measure().sandVolume;
        for (int step = 0; step < 8; ++step)
        {
            simulation.Step();
        }
        RequireNear(
            simulation.Measure().sandVolume,
            initialVolume,
            1.0e-13,
            "flat sand potential volume");
        Require(
            simulation.GetState().sand.velocity.GetMaxAbsComponent() <
                1.0e-11,
            "flat sand potential must not create velocity");
    }

    void TestSandAtRestAroundDryElevatedIsland()
    {
        // 只运行沙动力学，并关闭内力和相间动量交换。中央干地高于周围沙面，
        // 应当形成不可穿透接触面，而不能作为更高的沙重力势排斥静止沙层。
        SimulationConfig config;
        config.grid.resolution = {32, 24};
        config.grid.spacing = 0.05;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.enableSandInternalForce = false;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        SandSolver solver;
        SandStepReport report;
        state.SetZero();

        double constexpr sandSurface = 0.30;
        double constexpr islandTerrain = 0.40;
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            Vector2d const position = grid.GetCellGrid().PositionOf(cell);
            bool const isIsland =
                std::abs(position.x()) <= 0.10 &&
                std::abs(position.y()) <= 0.25;
            if (isIsland)
            {
                state.terrain[cell] = islandTerrain;
            }
            else
            {
                state.sand.height[cell] = sandSurface;
            }
        });

        solver.IntegrateDynamics(
            state, config, 0.005, boundary, scratch, report);
        RequireNear(
            state.sand.velocity.GetMaxAbsComponent(),
            0.0,
            1.0e-11,
            "dry elevated island must not repel a sand layer at rest");
    }

    void TestSandCanAdvanceOntoLowerDryGround()
    {
        // 与高地接触测试配对：右侧无沙但地形更低，因此沙重力势必须在界面
        // 产生向右速度。该对照防止后续实现把所有 sand/dry 面一律封死。
        SimulationConfig config;
        config.grid.resolution = {12, 8};
        config.grid.spacing = 0.1;
        config.physics.gravity = 10.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.enableSandInternalForce = false;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        SandSolver solver;
        SandStepReport report;
        state.SetZero();

        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (grid.GetCellGrid().PositionOf(cell).x() < 0.0)
            {
                state.sand.height[cell] = 0.30;
            }
        });

        double constexpr dt = 0.005;
        solver.IntegrateDynamics(
            state, config, dt, boundary, scratch, report);

        Vector2i const interfaceFace(
            config.grid.resolution.x() / 2,
            config.grid.resolution.y() / 2);
        double const expectedVelocity =
            config.physics.gravity * dt * 0.30 / config.grid.spacing;
        RequireNear(
            state.sand.velocity[0][interfaceFace],
            expectedVelocity,
            1.0e-13,
            "sand must remain able to advance onto lower dry ground");
    }

    void TestSandCannotAdvectIntoDryElevatedTerrain()
    {
        // 与水的旧速度回归对称：不施加重力/内力，仅检查已有沙速度是否会在
        // 本步力更新之前把质量送入高于沙面的干地。
        SimulationConfig config;
        config.grid.resolution = {12, 8};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.enableSandInternalForce = false;
        config.physics.sandTerrainContactTolerance = 0.005;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        SandSolver solver;
        SandStepReport report;
        state.SetZero();
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (grid.GetCellGrid().PositionOf(cell).x() < 0.0)
            {
                state.sand.height[cell] = 0.30;
            }
            else
            {
                state.terrain[cell] = 0.40;
            }
        });
        state.sand.velocity.SetConstant(Vector2d(0.20, 0.0));
        boundary.ApplyVelocityBoundary(state.sand.velocity);

        solver.IntegrateDynamics(
            state, config, 0.01, boundary, scratch, report);
        Vector2i const firstHighCell(
            config.grid.resolution.x() / 2,
            config.grid.resolution.y() / 2);
        Vector2i const interfaceFace = firstHighCell;
        RequireNear(
            state.sand.height[firstHighCell],
            0.0,
            0.0,
            "sand contact must block flux before height advection");
        RequireNear(
            state.sand.velocity[0][interfaceFace],
            0.0,
            0.0,
            "sand contact must clear stale normal velocity at a high wall");
    }

    void TestSandCanCrossSubToleranceTerrainStep()
    {
        // 与明显高台回归配对：干侧地形只比来沙表面高 0.004，小于配置的
        // 0.005 容差。已有法向速度应能携带沙质量穿过该面，证明容差只放宽
        // 微小重建起伏，而没有把所有 sand/dry 接触都封闭。
        SimulationConfig config;
        config.grid.resolution = {12, 8};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.enableSandInternalForce = false;
        config.physics.sandTerrainContactTolerance = 0.005;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        SandSolver solver;
        SandStepReport report;
        state.SetZero();
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (grid.GetCellGrid().PositionOf(cell).x() < 0.0)
            {
                state.sand.height[cell] = 0.30;
            }
            else
            {
                state.terrain[cell] = 0.304;
            }
        });
        state.sand.velocity.SetConstant(Vector2d(0.20, 0.0));
        boundary.ApplyVelocityBoundary(state.sand.velocity);

        solver.IntegrateDynamics(
            state, config, 0.01, boundary, scratch, report);
        Vector2i const firstRaisedCell(
            config.grid.resolution.x() / 2,
            config.grid.resolution.y() / 2);
        Require(
            state.sand.height[firstRaisedCell] > 0.0,
            "sand must cross a terrain step smaller than contact tolerance");
    }

    void TestSandCoulombFriction()
    {
        // 均匀 (3,4) 速度的模长为 5。干沙中 μg=2，dt=0.1 后模长应为
        // 4.8，两个分量统一乘 0.96；这样同时验证减速大小和方向不变。
        SimulationConfig config;
        config.grid.resolution = {10, 8};
        // 该测试刻意使用很大的 dt 检查解析摩擦减速；取 dx=1 可避免封闭
        // 边界附近的速度梯度让同一公开阶段随后执行的 F 更新发生无关翻转。
        config.grid.spacing = 1.0;
        config.physics.gravity = 10.0;
        config.physics.groundFriction = 0.2;
        config.physics.diffusionRate = 0.0;
        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        SandSolver solver;
        SandStepReport report;

        state.sand.height.SetConstant(1.0);
        state.sand.velocity.SetConstant(Vector2d(3.0, 4.0));
        boundary.ApplyVelocityBoundary(state.sand.velocity);
        solver.ApplyFrictionAndFinalize(
            state, config, 0.1, boundary, scratch, report);

        RequireNear(
            state.sand.velocity[0][Vector2i(4, 3)],
            2.88, 1.0e-13, "dry-sand friction x component");
        RequireNear(
            state.sand.velocity[1][Vector2i(4, 3)],
            3.84, 1.0e-13, "dry-sand friction y component");
        RequireNear(
            report.maxFrictionDeceleration,
            2.0, 1.0e-13, "dry-sand friction acceleration");

        // 相同沙柱完全含有 mixed water 时，浮力降低有效接触质量，减速必须变小。
        SimulationState wetState(grid);
        SimulationScratch wetScratch(grid);
        SandStepReport wetReport;
        wetState.sand.height.SetConstant(1.0);
        wetState.mixedWater.height.SetConstant(1.0);
        wetState.sand.velocity.SetConstant(Vector2d(3.0, 4.0));
        boundary.ApplyVelocityBoundary(wetState.sand.velocity);
        solver.ApplyFrictionAndFinalize(
            wetState, config, 0.1, boundary, wetScratch, wetReport);
        Require(
            wetReport.maxFrictionDeceleration <
                report.maxFrictionDeceleration,
            "mixed-water buoyancy must reduce ground friction");
        Require(
            wetState.sand.velocity[0][Vector2i(4, 3)] >
                state.sand.velocity[0][Vector2i(4, 3)],
            "wet sand should retain more speed under weaker friction");
    }

    void TestSandMixedWaterMomentumExchange()
    {
        // 取 rho_s*h_s=8、rho_w*h_m=2、K=c_e*h_m=400、dt=0.01。
        // 初始 d=0.4。论文显式模式给 J=1.6、d'=-0.6（发生过冲）；
        // 解析隐式模式给 d'=0.4/3.5，且两种模式都应保持面柱总动量。
        auto const runScheme =
            [](SandWaterExchangeScheme scheme,
               double expectedSandVelocity,
               double expectedMixedVelocity,
               bool expectOvershoot)
        {
            SimulationConfig config;
            config.grid.resolution = {16, 12};
            config.grid.spacing = 0.1;
            config.physics.gravity = 0.0;
            config.physics.sandDensity = 4.0;
            config.physics.waterDensity = 2.0;
            config.physics.diffusionRate = 0.0;
            config.physics.momentumExchangeRate = 400.0;
            config.physics.groundFriction = 0.0;
            config.physics.enableSandInternalForce = false;
            config.time.maxVelocity = 100.0;
            config.exchangeScheme = scheme;

            StaggeredGrid grid(
                config.grid.boundaryWidth,
                config.grid.spacing,
                config.grid.resolution);
            SimulationState state(grid);
            SimulationScratch scratch(grid);
            BoundaryConditions boundary(grid);
            SandSolver solver;
            SandStepReport report;

            state.sand.height.SetConstant(2.0);
            state.mixedWater.height.SetConstant(1.0);
            state.sand.velocity.SetConstant(Vector2d(0.3, 0.0));
            state.mixedWater.velocity.SetConstant(Vector2d(-0.1, 0.0));
            boundary.ApplyVelocityBoundary(state.sand.velocity);
            boundary.ApplyVelocityBoundary(state.mixedWater.velocity);

            double constexpr dt = 0.01;
            solver.IntegrateDynamics(
                state, config, dt, boundary, scratch, report);

            // 该面离边界和回溯路径均足够远，前面的常量场平流严格不改变值。
            Vector2i const probe(8, 5);
            RequireNear(
                state.sand.velocity[0][probe],
                expectedSandVelocity,
                2.0e-13,
                "sand velocity after drag exchange");
            RequireNear(
                state.mixedWater.velocity[0][probe],
                expectedMixedVelocity,
                2.0e-13,
                "mixed-water velocity after drag exchange");
            Require(
                report.exchangedFaceCount > 0,
                "wet sand must create exchange faces");
            Require(
                (report.exchangeOvershootFaceCount > 0) == expectOvershoot,
                "exchange overshoot diagnostic must match the scheme");
            Require(
                report.maxExchangeMomentumError < 2.0e-13,
                "exchange must conserve face-column momentum to roundoff");
            RequireNear(
                scratch.sandWaterImpulse[0][probe],
                8.0 * (0.3 - expectedSandVelocity),
                2.0e-13,
                "stored sand-to-water impulse");
        };

        runScheme(
            SandWaterExchangeScheme::PaperSequential,
            0.1,
            0.7,
            true);
        runScheme(
            SandWaterExchangeScheme::AnalyticImplicit,
            0.3 - (4.0 * (0.4 / 3.5)) / 8.0,
            -0.1 + (4.0 * (0.4 / 3.5)) / 2.0,
            false);

        // 回归 Simulation 的静止沙快速路径：即使沙速度为零，只要 mixed
        // water 在运动，M5 也必须进入沙子步并把一部分动量传给沙。
        SimulationConfig config;
        config.grid.resolution = {16, 12};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 400.0;
        config.physics.groundFriction = 0.0;
        config.physics.enableSandInternalForce = false;
        config.time.waterDt = 0.001;
        config.time.sandSubsteps = 1;
        config.exchangeScheme =
            SandWaterExchangeScheme::AnalyticImplicit;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 2.0; },
            [](Vector2d const &) { return 1.0; });
        simulation.GetState().mixedWater.velocity.SetConstant(
            Vector2d(0.2, 0.0));
        simulation.Step();
        Require(
            simulation.GetState().sand.velocity[0][Vector2i(8, 5)] >
                0.0,
            "moving mixed water must start a static sand layer");
        Require(
            simulation.GetLastSandStepReport().exchangedFaceCount > 0,
            "full Simulation step must report M5 exchange faces");
    }

    void TestStableSandSlope()
    {
        // |∂hs/∂x|=0.2 小于 μ=0.5。每个子步中柱重力产生的速度应被
        // Coulomb 摩擦完全消除。M6.6 进一步要求形变梯度读取摩擦后的最终
        // 速度；否则“步末静止”仍会在每步提交一次不可见的本构形变。
        SimulationConfig config;
        config.grid.resolution = {30, 16};
        config.grid.spacing = 0.04;
        config.physics.diffusionRate = 0.0;
        config.physics.groundFriction = 0.5;
        config.time.waterDt = 0.004;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return std::max(0.30 - 0.20 * std::abs(position.x()), 0.0);
            },
            [](Vector2d const &) { return 0.0; });

        double const initialVolume = simulation.Measure().sandVolume;
        for (int step = 0; step < 20; ++step)
        {
            simulation.Step();
        }

        RequireNear(
            simulation.Measure().sandVolume,
            initialVolume,
            2.0e-12,
            "stable sand slope mass");
        Require(
            simulation.GetState().sand.velocity.GetMaxAbsComponent() <
                1.0e-10,
            "sub-friction sand slope should remain at rest");
        double maximumDeformationChange = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell) &&
                simulation.GetState().sand.height[cell] >
                    config.time.heightEpsilon)
            {
                maximumDeformationChange = std::max(
                    maximumDeformationChange,
                    (simulation.GetState().sand.deformationGradient[cell] -
                     Matrix2d::Identity()).norm());
            }
        });
        RequireNear(
            maximumDeformationChange,
            0.0,
            1.0e-10,
            "static friction must prevent hidden deformation-gradient drift");

        // 以一半 dt 运行相同的 0.08 s 物理时间。两种离散都应满足同一个
        // 静摩擦不变量，而不是仅在某个偶然选定的步长下抵消漂移。
        config.time.waterDt = 0.002;
        Simulation refinedSimulation(config);
        refinedSimulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return std::max(0.30 - 0.20 * std::abs(position.x()), 0.0);
            },
            [](Vector2d const &) { return 0.0; });
        for (int step = 0; step < 40; ++step)
        {
            refinedSimulation.Step();
        }
        double refinedDeformationChange = 0.0;
        ForEach(refinedSimulation.GetGrid().GetCellGrid(),
            [&](Vector2i const &cell)
        {
            if (refinedSimulation.GetGrid().IsInsideCell(cell) &&
                refinedSimulation.GetState().sand.height[cell] >
                    config.time.heightEpsilon)
            {
                refinedDeformationChange = std::max(
                    refinedDeformationChange,
                    (refinedSimulation.GetState()
                         .sand.deformationGradient[cell] -
                     Matrix2d::Identity()).norm());
            }
        });
        RequireNear(
            refinedDeformationChange,
            0.0,
            1.0e-10,
            "half-dt static friction must prevent hidden F drift");
    }

    void TestDrySandPileShortRun()
    {
        // 陡峭 Gaussian 沙丘的坡度超过 μ，应该降低峰值并向外展宽，同时在
        // 封闭域中保持沙质量。这是 M3 的最小端到端运动场景。
        SimulationConfig config;
        config.grid.resolution = {28, 28};
        config.grid.spacing = 0.04;
        config.physics.diffusionRate = 0.0;
        config.physics.groundFriction = 0.42;
        config.time.waterDt = 0.003;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                double constexpr sigma = 0.12;
                return 0.34 * std::exp(
                    -position.squaredNorm() /
                    (2.0 * sigma * sigma));
            },
            [](Vector2d const &) { return 0.0; });

        double const initialVolume = simulation.Measure().sandVolume;
        double initialPeak = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell))
            {
                initialPeak = std::max(
                    initialPeak,
                    simulation.GetState().sand.height[cell]);
            }
        });

        for (int step = 0; step < 36; ++step)
        {
            simulation.Step();
        }

        double finalPeak = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell))
            {
                finalPeak = std::max(
                    finalPeak,
                    simulation.GetState().sand.height[cell]);
            }
        });
        ConservationReport const report = simulation.Measure();
        Require(
            finalPeak < initialPeak,
            "steep dry sand pile should lower its peak");
        RequireNear(
            report.sandVolume,
            initialVolume,
            3.0e-12,
            "dry sand pile closed-domain mass");
        Require(
            report.minHeight >= -1.0e-14,
            "dry sand pile height must remain non-negative");
        Require(
            simulation.GetLastSandStepReport().maxFrictionDeceleration > 0.0,
            "dry sand pile must exercise Coulomb friction");
        simulation.Validate("M3 short dry sand pile");
    }

    void TestWetCylinderThinFrontRegularization()
    {
        // phi=0.25 的陡直圆柱会在第一步生成约 1e-7 的薄前缘。未正则化的
        // 切向 div(h*sigma)/h 模板会在第二步放大到数千并使 F 翻转；默认
        // dx 比例下限必须让该最小复现场景在不依赖速度限幅的情况下通过。
        SimulationConfig config;
        config.grid.resolution = {32, 32};
        config.grid.spacing = 0.025;
        config.physics.diffusionRate = 0.0;
        config.physics.groundFriction = 0.42;
        config.time.waterDt = 0.002;
        config.time.sandSubsteps = 3;
        config.time.useAdaptiveDt = true;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return position.norm() <= 0.10 ? 0.22 : 0.0;
            },
            [](Vector2d const &position)
            {
                // w=s/3 对应 w/(w+s)=0.25。
                return position.norm() <= 0.10 ? 0.22 / 3.0 : 0.0;
            });

        for (int step = 0; step < 12; ++step)
        {
            simulation.Step();
        }
        SandStepReport const &report =
            simulation.GetLastSandStepReport();
        Require(
            report.velocityClampCount == 0,
            "regularized wet cylinder should not depend on velocity clamping");
        Require(
            report.maxInternalAcceleration < 1.0e3,
            "regularized wet-cylinder internal acceleration should stay bounded");
        Require(
            simulation.GetState().sand.velocity.GetMaxAbsComponent() < 2.0,
            "regularized wet-cylinder velocity should remain below the safety cap");
        SimulationScratch const &scratch = simulation.GetScratch();
        ForEach(scratch.internalAcceleration.GetGrids(),
            [&](int axis, Vector2i const &face)
        {
            RequireNear(
                scratch.internalAcceleration[axis][face],
                scratch.internalNormalAcceleration[axis][face] +
                    scratch.internalTangentialAcceleration[axis][face],
                1.0e-11,
                "internal-force diagnostic decomposition");
            Require(
                scratch.internalEffectiveHeight[axis][face] == 0.0 ||
                    scratch.internalEffectiveHeight[axis][face] + 1.0e-16 >=
                        scratch.internalFaceHeight[axis][face],
                "internal-force effective height must not be below face height");
        });
        ForEach(scratch.internalYieldResidual.GetGrid(), [&](Vector2i const &cell)
        {
            Require(
                std::isfinite(scratch.internalYieldResidual[cell]),
                "internal-stage yield residual must be finite");
        });
        simulation.Validate("M4 wet-cylinder thin-front regression");
    }

    void TestPostAdvectionConstitutiveProjection()
    {
        // 隔离本次算子时序：关闭重力、摩擦和内力，从一个明确越过干沙
        // Drucker-Prager 屈服面的均匀 trial F 出发，并跳过子步入口投影。
        // 开关开启后，h/P 平流完成处的返回映射应在 ComputeStress 前把它
        // 拉回屈服面，同时留下可审计的投影前/后诊断。
        SimulationConfig config;
        config.grid.resolution = {18, 14};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.groundFriction = 0.0;
        config.physics.enableSandInternalForce = false;
        config.physics.projectDeformationAfterAdvection = true;
        config.time.useAdaptiveDt = false;

        StaggeredGrid grid(
            config.grid.boundaryWidth,
            config.grid.spacing,
            config.grid.resolution);
        SimulationState state(grid);
        SimulationScratch scratch(grid);
        BoundaryConditions boundary(grid);
        SandSolver solver;
        SandStepReport report;
        state.SetZero();
        state.sand.height.SetConstant(0.30);

        Matrix2d trial = Matrix2d::Zero();
        trial(0, 0) = std::exp(0.15);
        trial(1, 1) = std::exp(-0.25);
        state.sand.deformationGradient.SetConstant(trial);
        state.sand.rotationAngle.SetZero();
        state.sand.stretch.SetConstant(trial);

        solver.IntegrateDynamics(
            state,
            config,
            0.001,
            boundary,
            scratch,
            report,
            false);

        Require(
            report.postAdvectionProjectionCount > 0,
            "post-advection projection must modify an inadmissible trial state");
        Require(
            report.maxPostAdvectionYieldResidualBefore > 1.0e-8,
            "pre-projection diagnostic must expose the positive yield residual");
        Require(
            report.maxPostAdvectionProjectionDelta > 0.0,
            "post-advection projection must report a nonzero F correction");
        Require(
            report.maxPostAdvectionYieldResidualAfter <= 1.0e-7,
            "post-advection projection must return F to the yield surface");
        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (scratch.sandActive[cell])
            {
                Require(
                    scratch.postAdvectionYieldResidualAfter[cell] <= 1.0e-7,
                    "projected cell residual must be admissible");
                Require(
                    scratch.internalYieldResidual[cell] <= 1.0e-7,
                    "ComputeStress must observe the projected F");
            }
        });
    }

    void TestFrozenBoundSaturationTransport()
    {
        // FrozenBound 必须让初始混合水以与沙高相同的守恒通量平流。这里主动
        // 设置非零沙速度，同时保留本来会触发水扩散/交换的参数，确认诊断
        // 分支既保持局部含水比，也确实没有运行独立水动力学。
        SimulationConfig config;
        config.grid.resolution = {24, 16};
        config.grid.spacing = 0.05;
        config.physics.saturationMode =
            SaturationEvolutionMode::FrozenBound;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.25;
        config.physics.momentumExchangeRate = 5000.0;
        config.physics.groundFriction = 0.0;
        config.physics.enableSandInternalForce = false;
        config.time.waterDt = 0.005;
        config.time.sandSubsteps = 1;
        config.time.useAdaptiveDt = true;

        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return position.x() < 0.15 &&
                    std::abs(position.y()) < 0.20
                    ? 0.20
                    : 0.0;
            },
            [](Vector2d const &position)
            {
                return position.x() < 0.15 &&
                    std::abs(position.y()) < 0.20
                    ? 0.10
                    : 0.0;
            });

        ForEach(simulation.GetGrid().GetFaceGrids(),
            [&](int axis, Vector2i const &face)
        {
            if (axis == 0 &&
                !simulation.GetGrid().IsBoundaryFace(axis, face))
            {
                simulation.GetState().sand.velocity[axis][face] = 0.20;
            }
        });
        double const initialSand = simulation.Measure().sandVolume;
        double const initialWater = simulation.Measure().totalWaterVolume;
        simulation.Step();

        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (!simulation.GetGrid().IsInsideCell(cell))
            {
                return;
            }
            double const sand = simulation.GetState().sand.height[cell];
            double const mixed =
                simulation.GetState().mixedWater.height[cell];
            RequireNear(
                simulation.GetState().pureWater.height[cell],
                0.0, 1.0e-14, "frozen-bound pure-water height");
            if (sand > 1.0e-10)
            {
                RequireNear(
                    mixed / sand,
                    0.5, 1.0e-11,
                    "frozen-bound mixed-water to sand ratio");
            }
        });
        RequireNear(
            simulation.Measure().sandVolume,
            initialSand, 1.0e-12, "frozen-bound sand conservation");
        RequireNear(
            simulation.Measure().totalWaterVolume,
            initialWater, 1.0e-12, "frozen-bound water conservation");
        Require(
            simulation.GetLastWaterStepReport().DiffusionVolumeChange() == 0.0 &&
                simulation.GetLastSandStepReport().exchangedFaceCount == 0,
            "frozen-bound mode should skip diffusion and momentum exchange");
        ForEach(simulation.GetGrid().GetFaceGrids(),
            [&](int axis, Vector2i const &face)
        {
            RequireNear(
                simulation.GetState().mixedWater.velocity[axis][face],
                simulation.GetState().sand.velocity[axis][face],
                1.0e-14,
                "frozen-bound water velocity should match sand velocity");
        });
        simulation.Validate("frozen-bound saturation transport regression");
    }

    void TestAdvancingDryFrontVelocityExtrapolation()
    {
        // 无重力、常速度矩形水柱应近似平移。该测试专门防止“新湿面沿用旧零速”
        // 的回归：若速度没有从旧湿区外推，前缘会停住并形成远高于初值的尖峰。
        SimulationConfig config;
        config.grid.resolution = {42, 14};
        config.grid.spacing = 0.05;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.time.waterDt = 0.01;
        config.time.useAdaptiveDt = false;
        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &position)
            {
                return position.x() < -0.2 ? 0.2 : 0.0;
            });

        ForEach(simulation.GetGrid().GetFaceGrids(),
            [&](int axis, Vector2i const &face)
        {
            if (axis == 0 &&
                !simulation.GetGrid().IsBoundaryFace(axis, face))
            {
                simulation.GetState().pureWater.velocity[axis][face] = 0.25;
            }
        });

        double initialMoment = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell))
            {
                initialMoment +=
                    simulation.GetState().pureWater.height[cell] *
                    simulation.GetGrid().GetCellGrid().PositionOf(cell).x();
            }
        });

        for (int step = 0; step < 40; ++step)
        {
            simulation.Step();
        }

        double finalMoment = 0.0;
        double maximumHeight = 0.0;
        ForEach(simulation.GetGrid().GetCellGrid(), [&](Vector2i const &cell)
        {
            if (simulation.GetGrid().IsInsideCell(cell))
            {
                double const height =
                    simulation.GetState().pureWater.height[cell];
                finalMoment +=
                    height *
                    simulation.GetGrid().GetCellGrid().PositionOf(cell).x();
                maximumHeight = std::max(maximumHeight, height);
            }
        });
        Require(
            finalMoment > initialMoment,
            "dry-front water column must advance in its velocity direction");
        Require(
            maximumHeight <= 0.2 + 1.0e-12,
            "newly wet faces must not create a stalled height spike");
    }

    void TestConstitutiveSaturationLowpass()
    {
        SimulationConfig config;
        config.grid.resolution = {12, 10};
        config.grid.spacing = 0.1;
        config.physics.gravity = 0.0;
        config.physics.diffusionRate = 0.0;
        config.physics.momentumExchangeRate = 0.0;
        config.physics.groundFriction = 0.0;
        config.physics.constitutiveSaturationLowpass = true;
        config.time.waterDt = 1.0e-4;
        config.time.sandSubsteps = 1;
        config.time.useAdaptiveDt = false;

        Simulation simulation(config);
        simulation.InitializeHeights(
            [](Vector2d const &) { return 0.0; },
            [](Vector2d const &) { return 0.5; },
            [](Vector2d const &position)
            {
                return position.x() < 0.0 ? 0.1 : 0.4;
            });
        simulation.Step();

        SimulationScratch const &scratch = simulation.GetScratch();
        Grid const &grid = simulation.GetGrid().GetCellGrid();
        bool changedByFilter = false;
        ForEach(grid, [&](Vector2i const &cell)
        {
            if (!scratch.sandActive[cell])
            {
                return;
            }
            double weighted = 0.0;
            double weights = 0.0;
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    Vector2i const neighbor = cell + Vector2i(dx, dy);
                    if (!grid.IsValid(neighbor) ||
                        !scratch.sandActive[neighbor])
                    {
                        continue;
                    }
                    double const weight =
                        (dx == 0 ? 2.0 : 1.0) *
                        (dy == 0 ? 2.0 : 1.0);
                    weighted += weight *
                        scratch.rawConstitutiveSaturation[neighbor];
                    weights += weight;
                }
            }
            double const expected = weighted / weights;
            RequireNear(
                scratch.saturation[cell], expected, 1.0e-13,
                "constitutive saturation should use masked binomial lowpass");
            changedByFilter = changedByFilter ||
                std::abs(
                    scratch.saturation[cell] -
                    scratch.rawConstitutiveSaturation[cell]) > 1.0e-3;
        });
        Require(
            changedByFilter,
            "non-uniform saturation should be changed by constitutive lowpass");
    }

    struct TestCase
    {
        /// runner 中的测试描述、筛选分组和实际测试函数。
        std::string_view name;
        std::string_view group;
        std::function<void()> run;
    };
}

int main(int argc, char **argv)
{
    // --group unit|integration 供构建流水线分层执行；缺省 all 执行全部测试。
    std::string_view selectedGroup = "all";
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (std::string_view(argv[i]) == "--group")
        {
            selectedGroup = argv[i + 1];
        }
    }

    std::vector<TestCase> const tests = {
        {"config validation", "unit", TestConfigValidation},
        {"dry Drucker-Prager closed form", "unit",
         TestDryDruckerPragerClosedFormProjection},
        {"constitutive projection idempotence", "unit",
         TestConstitutiveProjectionIdempotence},
        {"adaptive sand elastic CFL", "integration",
         TestAdaptiveSandElasticCfl},
        {"state initialization", "unit", TestStateInitialization},
        {"boundary conditions", "unit", TestBoundaryConditions},
        {"masks and diagnostics", "unit", TestMasksAndDiagnostics},
        {"simulation initialization", "integration", TestSimulationInitialization},
        {"NPY output", "integration", TestNpyOutput},
        {"M6.5 maximum time step", "unit", TestM65MaximumStep},
        {"M6 scene sources", "unit", TestSceneSources},
        {"M6 scene loader and scheduling", "integration",
         TestSceneLoaderAndScheduling},
        {"MAC operators", "unit", TestMacOperators},
        {"finite-volume height advection", "unit", TestFiniteVolumeHeightAdvection},
        {"semi-Lagrangian zero flow", "unit", TestSemiLagrangianZeroFlow},
        {"mixed-water diffusion", "unit", TestMixedWaterDiffusion},
        {"water layer transfer", "unit", TestWaterLayerTransfer},
        {"lake at rest", "integration", TestLakeAtRest},
        {"lake at rest around dry elevated island", "integration",
         TestLakeAtRestAroundDryElevatedIsland},
        {"two-layer pressure across pure/mixed partition", "integration",
         TestTwoLayerPressureAcrossPureMixedPartition},
        {"mixed water at rest around dry elevated island", "integration",
         TestMixedWaterAtRestAroundDryElevatedIsland},
        {"water cannot advect into dry elevated terrain", "integration",
         TestWaterCannotAdvectIntoDryElevatedTerrain},
        {"M1 short dam break", "integration", TestDamBreakShortRun},
        {"M2 short two-layer water", "integration", TestTwoLayerWaterShortRun},
        {"advancing dry-front velocity extrapolation", "integration",
         TestAdvancingDryFrontVelocityExtrapolation},
        {"sand potential at rest", "integration", TestSandPotentialAtRest},
        {"sand at rest around dry elevated island", "integration",
         TestSandAtRestAroundDryElevatedIsland},
        {"sand advances onto lower dry ground", "integration",
         TestSandCanAdvanceOntoLowerDryGround},
        {"sand cannot advect into dry elevated terrain", "integration",
         TestSandCannotAdvectIntoDryElevatedTerrain},
        {"sand crosses sub-tolerance terrain step", "integration",
         TestSandCanCrossSubToleranceTerrainStep},
        {"sand Coulomb friction", "unit", TestSandCoulombFriction},
        {"sand-mixed-water momentum exchange", "unit",
         TestSandMixedWaterMomentumExchange},
        {"stable sand slope", "integration", TestStableSandSlope},
        {"M3 short dry sand pile", "integration", TestDrySandPileShortRun},
        {"M4 wet-cylinder thin-front regularization", "integration",
         TestWetCylinderThinFrontRegularization},
        {"post-advection constitutive projection", "integration",
         TestPostAdvectionConstitutiveProjection},
        {"frozen-bound saturation transport", "integration",
         TestFrozenBoundSaturationTransport},
        {"constitutive saturation lowpass", "integration",
         TestConstitutiveSaturationLowpass},
    };

    int failures = 0;
    int executed = 0;
    for (TestCase const &test : tests)
    {
        if (selectedGroup != "all" && selectedGroup != test.group)
        {
            continue;
        }
        ++executed;
        try
        {
            test.run();
            fmt::print("[PASS] {} ({})\n", test.name, test.group);
        }
        catch (std::exception const &error)
        {
            ++failures;
            fmt::print(stderr, "[FAIL] {} ({}): {}\n", test.name, test.group, error.what());
        }
    }

    fmt::print("{} tests executed, {} failures\n", executed, failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
