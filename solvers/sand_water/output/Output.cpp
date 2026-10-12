#include "output/Output.h"
#include "physics/SandConstitutive.h"

#include <npy.hpp>

namespace Pivot
{
    namespace
    {
        using Shape = std::vector<unsigned long>;

        template <typename Type>
        void WriteArray(
            std::filesystem::path const &path,
            Shape const &shape,
            std::vector<Type> const &data)
        {
            npy::SaveArrayAsNumpy(
                path.string(), false, static_cast<unsigned int>(shape.size()),
                shape.data(), data);
        }

        void WriteScalarField(
            std::filesystem::path const &path,
            GridData<double> const &field)
        {
            Vector2i const size = field.GetGrid().GetSize();
            WriteArray(path, {
                static_cast<unsigned long>(size.x()),
                static_cast<unsigned long>(size.y())}, field.GetData());
        }

        void WriteVelocityField(
            std::filesystem::path const &directory,
            std::string_view prefix,
            SGridData<double> const &velocity)
        {
            for (int axis = 0; axis < 2; ++axis)
            {
                // 两个 MAC 分量尺寸不同，必须保存为独立数组，不能拼成一个
                // 看似规则但丢失网格拓扑信息的向量数组。
                Vector2i const size = velocity[axis].GetGrid().GetSize();
                WriteArray(
                    directory / fmt::format("{}_velocity_{}.npy", prefix, axis == 0 ? "x" : "y"),
                    {
                        static_cast<unsigned long>(size.x()),
                        static_cast<unsigned long>(size.y())
                    },
                    velocity[axis].GetData());
            }
        }

        void WriteMatrixField(
            std::filesystem::path const &path,
            GridData<Matrix2d> const &field)
        {
            Vector2i const size = field.GetGrid().GetSize();
            std::vector<double> values;
            values.reserve(static_cast<std::size_t>(size.prod()) * 4);
            // 输出形状为 [nx, ny, 2, 2]，最后两个维度按行主序写矩阵元素。
            // 这使 Python 可直接对最后两个轴调用 det/eig 等 NumPy 算子。
            for (int i = 0; i < size.x(); ++i)
            {
                for (int j = 0; j < size.y(); ++j)
                {
                    Matrix2d const &matrix = field[Vector2i(i, j)];
                    for (int row = 0; row < 2; ++row)
                    {
                        for (int col = 0; col < 2; ++col)
                        {
                            values.push_back(matrix(row, col));
                        }
                    }
                }
            }
            WriteArray(path, {
                static_cast<unsigned long>(size.x()),
                static_cast<unsigned long>(size.y()),
                2UL,
                2UL}, values);
        }

        GridData<double> ComputeSaturation(SimulationState const &state)
        {
            // 这是供 M0 概览图使用的“水在总材料柱中的比例”，不是最终论文
            // 本构方程中的孔隙饱和度；后者将在耦合模块中单独定义。
            GridData<double> saturation(state.terrain.GetGrid());
            ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
            {
                double const water =
                    state.pureWater.height[cell] + state.mixedWater.height[cell];
                double const total = water + state.sand.height[cell];
                saturation[cell] = total > 0.0 ? water / total : 0.0;
            });
            return saturation;
        }

        GridData<double> ComputeDeterminant(SimulationState const &state)
        {
            GridData<double> determinant(state.terrain.GetGrid(), 1.0);
            ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
            {
                determinant[cell] = state.sand.deformationGradient[cell].determinant();
            });
            return determinant;
        }

        void ComputeConstitutiveDiagnostics(
            SimulationState const &state,
            SimulationConfig const &config,
            GridData<Matrix2d> &stress,
            GridData<double> &yieldResidual)
        {
            SandConstitutive const constitutive(
                config.physics,
                config.time.determinantEpsilon);
            ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
            {
                double const sandHeight = state.sand.height[cell];
                if (sandHeight <= config.time.heightEpsilon)
                {
                    stress[cell] = Matrix2d::Zero();
                    yieldResidual[cell] = 0.0;
                    return;
                }
                double const water =
                    state.pureWater.height[cell] +
                    state.mixedWater.height[cell];
                double const saturation =
                    water / (water + sandHeight);
                stress[cell] = constitutive.ComputeCauchyStress(
                    state.sand.deformationGradient[cell],
                    saturation);
                yieldResidual[cell] =
                    constitutive.YieldResidual(
                        stress[cell], saturation);
            });
        }

        void AppendMetrics(
            std::filesystem::path const &runDirectory,
            SimulationSnapshot const &snapshot,
            ConservationReport const &report)
        {
            std::ofstream output(runDirectory / "metrics.csv", std::ios::app);
            if (!output)
            {
                throw std::runtime_error("failed to append metrics.csv");
            }
            output << std::setprecision(17)
                   // 使用足够精度保留 double 往返精度，便于做严格回归比较。
                   << snapshot.stepIndex << ',' << snapshot.time << ','
                   << report.sandVolume << ','
                   << report.pureWaterVolume << ','
                   << report.mixedWaterVolume << ','
                   << report.totalWaterVolume << ','
                   << report.horizontalMomentum.x() << ','
                   << report.horizontalMomentum.y() << ','
                   << report.minHeight << ','
                   << report.minDetF << ','
                   << report.maxYieldResidual << ','
                   << report.maxVelocity << ','
                   << report.timeStep << ','
                   << report.cfl << ','
                   << report.mixedDiffusionVolumeChange << ','
                   << report.waterAdvectionVolumeChange << ','
                   << report.layerCorrectionVolumeChange << ','
                   << report.maxLayerTransfer << ','
                   << report.maxLayerMomentumError << ','
                   << report.sandAdvectionVolumeChange << ','
                   << report.maxSandGravityAcceleration << ','
                   << report.maxExchangeRelativeSpeedBefore << ','
                   << report.maxExchangeRelativeSpeedAfter << ','
                   << report.maxExchangeImpulse << ','
                   << report.maxExchangeMomentumError << ','
                   << report.exchangedFaceCount << ','
                   << report.exchangeOvershootFaceCount << ','
                   << report.maxSandInternalAcceleration << ','
                   << report.sandHeightAtMaxInternalAcceleration << ','
                   << report.maxSandNormalStressAcceleration << ','
                   << report.maxSandTangentialStressAcceleration << ','
                   << report.maxPostAdvectionYieldResidualBefore << ','
                   << report.maxPostAdvectionYieldResidualAfter << ','
                   << report.maxPostAdvectionProjectionDelta << ','
                   << report.postAdvectionProjectionCount << ','
                   << report.maxSandVelocityBeforeClamping << ','
                   << report.sandVelocityClampCount << ','
                   << report.maxSandFrictionDeceleration << ','
                   << report.frictionStoppedFaces << ','
                   << report.sandCfl << '\n';
        }

        std::string_view SnapshotLevelName(SnapshotLevel level)
        {
            switch (level)
            {
            case SnapshotLevel::Render:
                return "render";
            case SnapshotLevel::Basic:
                return "basic";
            case SnapshotLevel::Diagnostic:
                return "diagnostic";
            }
            throw std::logic_error("unknown snapshot level");
        }
    }

    void FrameExporter::WriteMetadata(
        SimulationConfig const &config,
        std::filesystem::path const &runDirectory,
        std::string_view sceneName,
        SnapshotLevel level) const
    {
        std::filesystem::create_directories(runDirectory / "frames");
        std::filesystem::create_directories(runDirectory / "previews");

        // metadata 是一次运行内不随帧改变的自描述信息。format_version 用于
        // 将来字段升级时让读取工具显式选择兼容路径。
        YAML::Node root;
        // M6.5 新增可省略大数组的 Render 帧，因此记录输出级别供外部工具
        // 在请求速度/本构面板前判断字段是否存在。
        root["format_version"] = 8;
        root["stage"] = "M6.5";
        root["output"]["snapshot_level"] =
            std::string(SnapshotLevelName(level));
        if (!sceneName.empty())
        {
            root["scene"] = std::string(sceneName);
        }
        root["physics"]["lame_mu"] = config.physics.lameMu;
        root["physics"]["lame_lambda"] = config.physics.lameLambda;
        root["grid"]["resolution"] = config.grid.resolution;
        root["grid"]["spacing"] = config.grid.spacing;
        root["grid"]["boundary_width"] = config.grid.boundaryWidth;
        root["grid"]["center"] = config.grid.center;
        root["physics"]["gravity"] = config.physics.gravity;
        root["physics"]["sand_density"] = config.physics.sandDensity;
        root["physics"]["water_density"] = config.physics.waterDensity;
        root["physics"]["saturation_mode"] =
            config.physics.saturationMode == SaturationEvolutionMode::Dynamic
            ? "dynamic"
            : "frozen_bound";
        root["physics"]["diffusion_rate"] = config.physics.diffusionRate;
        root["physics"]["momentum_exchange_rate"] =
            config.physics.momentumExchangeRate;
        root["physics"]["momentum_exchange_scheme"] =
            config.exchangeScheme == SandWaterExchangeScheme::PaperSequential
            ? "paper_sequential"
            : "analytic_implicit";
        root["physics"]["ground_friction"] =
            config.physics.groundFriction;
        root["physics"]["sand_terrain_contact_tolerance"] =
            config.physics.sandTerrainContactTolerance;
        root["physics"]["sand_internal_force"] =
            config.physics.enableSandInternalForce;
        root["physics"]["project_deformation_after_advection"] =
            config.physics.projectDeformationAfterAdvection;
        root["physics"]["drucker_prager_friction"] =
            config.physics.druckerPragerFriction;
        root["physics"]["cohesion"]["c0"] =
            config.physics.cohesion.c0;
        root["physics"]["cohesion"]["c1"] =
            config.physics.cohesion.c1;
        root["physics"]["cohesion"]["c2"] =
            config.physics.cohesion.c2;
        root["physics"]["cohesion"]["phi1"] =
            config.physics.cohesion.phi1;
        root["physics"]["cohesion"]["phi2"] =
            config.physics.cohesion.phi2;
        root["physics"]["cohesion"]["phi3"] =
            config.physics.cohesion.phi3;
        root["physics"]["water_layer_transfer"] = "donor_momentum_recombination";
        root["time"]["water_dt"] = config.time.waterDt;
        root["time"]["sand_substeps"] = config.time.sandSubsteps;
        root["time"]["adaptive"] = config.time.useAdaptiveDt;
        root["time"]["target_cfl"] = config.time.targetCfl;
        root["time"]["height_epsilon"] = config.time.heightEpsilon;
        root["time"]["internal_force_height_floor"] =
            config.time.internalForceHeightFloor;
        root["time"]["internal_force_height_fraction"] =
            config.time.internalForceHeightFraction;
        root["time"]["determinant_epsilon"] = config.time.determinantEpsilon;
        root["time"]["maximum_velocity"] = config.time.maxVelocity;

        std::ofstream metadata(runDirectory / "metadata.yaml");
        if (!metadata)
        {
            throw std::runtime_error("failed to write metadata.yaml");
        }
        metadata << root << '\n';

        std::ofstream metrics(runDirectory / "metrics.csv", std::ios::trunc);
        if (!metrics)
        {
            throw std::runtime_error("failed to write metrics.csv");
        }
        metrics
            << "step,time,sand_volume,pure_water_volume,mixed_water_volume,"
               "total_water_volume,momentum_x,momentum_y,min_height,min_det_f,"
               "max_yield_residual,max_velocity,dt,cfl,"
               "mixed_diffusion_volume_change,water_advection_volume_change,"
               "layer_correction_volume_change,max_layer_transfer,"
               "max_layer_momentum_error,sand_advection_volume_change,"
               "max_sand_gravity_acceleration,"
               "max_exchange_relative_speed_before,"
               "max_exchange_relative_speed_after,max_exchange_impulse,"
               "max_exchange_momentum_error,exchanged_face_count,"
               "exchange_overshoot_face_count,"
               "max_sand_internal_acceleration,"
               "sand_height_at_max_internal_acceleration,"
               "max_sand_normal_stress_acceleration,"
               "max_sand_tangential_stress_acceleration,"
               "max_post_advection_yield_residual_before,"
               "max_post_advection_yield_residual_after,"
               "max_post_advection_projection_delta,"
               "post_advection_projection_count,"
               "max_sand_velocity_before_clamping,sand_velocity_clamp_count,"
               "max_sand_friction_deceleration,"
               "friction_stopped_faces,sand_cfl\n";
    }

    std::filesystem::path FrameExporter::WriteFrame(
        SimulationSnapshot const &snapshot,
        ConservationReport const &report,
        SimulationConfig const &config,
        SnapshotLevel level,
        std::filesystem::path const &runDirectory,
        bool appendMetrics) const
    {
        std::filesystem::path const frameDirectory =
            // 固定宽度编号保证目录的字典序就是时间顺序。
            runDirectory / "frames" / fmt::format("{:06d}", snapshot.stepIndex);
        std::filesystem::create_directories(frameDirectory);

        SimulationState const &state = snapshot.state;
        WriteScalarField(frameDirectory / "terrain.npy", state.terrain);
        WriteScalarField(frameDirectory / "sand_height.npy", state.sand.height);
        WriteScalarField(frameDirectory / "pure_water_height.npy", state.pureWater.height);
        WriteScalarField(frameDirectory / "mixed_water_height.npy", state.mixedWater.height);
        GridData<double> totalWater(state.terrain.GetGrid());
        ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
        {
            totalWater[cell] =
                state.pureWater.height[cell] + state.mixedWater.height[cell];
        });
        WriteScalarField(frameDirectory / "total_water.npy", totalWater);
        if (level != SnapshotLevel::Render)
        {
            WriteScalarField(
                frameDirectory / "saturation.npy",
                ComputeSaturation(state));
            if (snapshot.scratch != nullptr)
            {
                // 正值表示 pure -> mixed，负值表示 mixed -> pure；该场是定位
                // 层转换方向和沙容量边界最直接的可视化量。
                WriteScalarField(
                    frameDirectory / "mixed_height_delta.npy",
                    snapshot.scratch->mixedHeightDelta);
            }

            // Render 级别不写六张 MAC 面速度；surface 模式只依赖高度。
            WriteVelocityField(
                frameDirectory, "sand", state.sand.velocity);
            WriteVelocityField(
                frameDirectory, "pure_water",
                state.pureWater.velocity);
            WriteVelocityField(
                frameDirectory, "mixed_water",
                state.mixedWater.velocity);
        }

        if (level == SnapshotLevel::Diagnostic)
        {
            // 本构数据体积较大，只在诊断/回归运行中输出。stress/yield
            // 由当前 F 重新计算，避免依赖 scratch 在最后一个子阶段的暂存值。
            WriteMatrixField(
                frameDirectory / "deformation_gradient.npy",
                state.sand.deformationGradient);
            WriteScalarField(frameDirectory / "det_f.npy", ComputeDeterminant(state));
            WriteScalarField(frameDirectory / "rotation_angle.npy", state.sand.rotationAngle);
            WriteMatrixField(frameDirectory / "stretch.npy", state.sand.stretch);
            GridData<Matrix2d> stress(
                state.terrain.GetGrid(), Matrix2d::Zero());
            GridData<double> yieldResidual(state.terrain.GetGrid());
            ComputeConstitutiveDiagnostics(
                state, config, stress, yieldResidual);
            WriteMatrixField(
                frameDirectory / "stress.npy", stress);
            WriteScalarField(
                frameDirectory / "yield_residual.npy",
                yieldResidual);
            if (snapshot.scratch != nullptr)
            {
                // 这些场来自最近一次真正的 ApplyInternalForce，而不是帧末重算。
                // 它们用于区分法向/切向离散、薄层分母和越界 trial F 谁最先
                // 产生网格对齐的条纹。
                WriteMatrixField(
                    frameDirectory / "internal_stress.npy",
                    snapshot.scratch->stress);
                WriteMatrixField(
                    frameDirectory / "internal_weighted_stress.npy",
                    snapshot.scratch->weightedStress);
                WriteScalarField(
                    frameDirectory / "internal_saturation.npy",
                    snapshot.scratch->saturation);
                WriteScalarField(
                    frameDirectory / "internal_saturation_raw.npy",
                    snapshot.scratch->rawConstitutiveSaturation);
                WriteScalarField(
                    frameDirectory / "internal_yield_residual.npy",
                    snapshot.scratch->internalYieldResidual);
                WriteScalarField(
                    frameDirectory /
                        "post_advection_yield_residual_before.npy",
                    snapshot.scratch->postAdvectionYieldResidualBefore);
                WriteScalarField(
                    frameDirectory /
                        "post_advection_yield_residual_after.npy",
                    snapshot.scratch->postAdvectionYieldResidualAfter);
                WriteScalarField(
                    frameDirectory /
                        "post_advection_projection_delta.npy",
                    snapshot.scratch->postAdvectionProjectionDelta);
                WriteVelocityField(
                    frameDirectory, "internal_total_acceleration",
                    snapshot.scratch->internalAcceleration);
                WriteVelocityField(
                    frameDirectory, "internal_normal_acceleration",
                    snapshot.scratch->internalNormalAcceleration);
                WriteVelocityField(
                    frameDirectory, "internal_tangential_acceleration",
                    snapshot.scratch->internalTangentialAcceleration);
                WriteVelocityField(
                    frameDirectory, "internal_face_height",
                    snapshot.scratch->internalFaceHeight);
                WriteVelocityField(
                    frameDirectory, "internal_effective_height",
                    snapshot.scratch->internalEffectiveHeight);
            }
        }

        if (appendMetrics)
        {
            AppendMetrics(runDirectory, snapshot, report);
        }
        return frameDirectory;
    }

    void FrameExporter::WriteMetrics(
        SimulationSnapshot const &snapshot,
        ConservationReport const &report,
        std::filesystem::path const &runDirectory) const
    {
        AppendMetrics(runDirectory, snapshot, report);
    }

    FrameExporter::HeightMesh FrameExporter::BuildHeightMesh(
        GridData<double> const &terrain,
        GridData<double> const *firstHeight,
        GridData<double> const *secondHeight,
        int boundaryWidth)
    {
        Grid const &grid = terrain.GetGrid();
        Vector2i const fullSize = grid.GetSize();
        Vector2i const meshSize =
            fullSize - Vector2i::Constant(2 * boundaryWidth);
        if ((meshSize.array() < 2).any())
        {
            throw std::invalid_argument(
                "OBJ height mesh requires at least 2x2 interior cells");
        }

        HeightMesh mesh;
        mesh.vertices.reserve(
            static_cast<std::size_t>(meshSize.prod()));
        for (int i = 0; i < meshSize.x(); ++i)
        {
            for (int j = 0; j < meshSize.y(); ++j)
            {
                Vector2i const cell =
                    Vector2i(i, j) +
                    Vector2i::Constant(boundaryWidth);
                Vector2d const position = grid.PositionOf(cell);
                double height = terrain[cell];
                if (firstHeight != nullptr)
                {
                    height += (*firstHeight)[cell];
                }
                if (secondHeight != nullptr)
                {
                    height += (*secondHeight)[cell];
                }
                mesh.vertices.emplace_back(
                    position.x(), position.y(), height);
            }
        }

        // 每个规则四边形沿同一对角线拆成两个逆时针三角形。索引函数必须
        // 与 GridData 的 x-major/y-minor 顺序一致，便于 OBJ 与 NPY 对照。
        auto const indexOf = [&](int i, int j)
        {
            return j + meshSize.y() * i;
        };
        mesh.faces.reserve(
            static_cast<std::size_t>(
                2 * (meshSize.x() - 1) *
                    (meshSize.y() - 1)));
        for (int i = 0; i + 1 < meshSize.x(); ++i)
        {
            for (int j = 0; j + 1 < meshSize.y(); ++j)
            {
                int const v00 = indexOf(i, j);
                int const v10 = indexOf(i + 1, j);
                int const v01 = indexOf(i, j + 1);
                int const v11 = indexOf(i + 1, j + 1);
                mesh.faces.emplace_back(v00, v10, v11);
                mesh.faces.emplace_back(v00, v11, v01);
            }
        }
        return mesh;
    }

    void FrameExporter::WriteObj(
        std::filesystem::path const &path,
        HeightMesh const &mesh)
    {
        std::ofstream output(path, std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error(fmt::format(
                "failed to write OBJ '{}'", path.string()));
        }
        output << std::setprecision(17);
        for (Vector3d const &vertex : mesh.vertices)
        {
            output << "v " << vertex.x() << ' '
                   << vertex.y() << ' '
                   << vertex.z() << '\n';
        }
        for (Vector3i const &face : mesh.faces)
        {
            // OBJ 索引从 1 开始。
            output << "f " << face.x() + 1 << ' '
                   << face.y() + 1 << ' '
                   << face.z() + 1 << '\n';
        }
    }

    void FrameExporter::WriteObjMeshes(
        SimulationSnapshot const &snapshot,
        SimulationConfig const &config,
        std::filesystem::path const &directory) const
    {
        std::filesystem::create_directories(directory);
        SimulationState const &state = snapshot.state;
        int const boundaryWidth =
            config.grid.boundaryWidth;
        WriteObj(
            directory / "terrain.obj",
            BuildHeightMesh(
                state.terrain, nullptr, nullptr,
                boundaryWidth));
        WriteObj(
            directory / "sand_surface.obj",
            BuildHeightMesh(
                state.terrain, &state.sand.height, nullptr,
                boundaryWidth));
        WriteObj(
            directory / "water_surface.obj",
            BuildHeightMesh(
                state.terrain,
                &state.mixedWater.height,
                &state.pureWater.height,
                boundaryWidth));
    }
}
