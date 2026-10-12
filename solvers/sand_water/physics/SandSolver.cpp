#include "physics/SandSolver.h"

namespace Pivot
{
    // ----- M4 sand-substep orchestration ----------------------------------------

    void SandSolver::IntegrateDynamics(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions &boundary,
        SimulationScratch &scratch,
        SandStepReport &report,
        bool projectInputState) const
    {
        if (!(dt > 0.0) || !std::isfinite(dt))
        {
            throw std::invalid_argument(
                "sand time step must be finite and positive");
        }

        boundary.ApplyHeightBoundary(state.sand.height);
        boundary.ApplyVelocityBoundary(state.sand.velocity);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        boundary.BuildSandTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            config.physics.sandTerrainContactTolerance,
            scratch.sandTerrainFaceOpen);
        // 沙高使用当前面速度先做守恒平流，所以接触约束必须早于重力势更新。
        // 这可防止上一子步残留法向速度在被清理前把沙质量送入高地单元。
        boundary.ApplyTerrainContactBoundary(
            state.sand.velocity, scratch.sandTerrainFaceOpen);

        // 每个外层水步的首个沙子步必须按最新含水量重新投影。后续子步的输入
        // 已在上一子步末完成同一投影，M7 可跳过该幂等重复工作；公开接口默认
        // 仍投影，避免直接调用 Solver 的测试或其他前端依赖隐藏前置条件。
        if (projectInputState)
        {
            ProjectDeformationGradient(state, config, scratch);
        }

        // 高度更新前的面掩码表示真正拥有 n 时刻沙速度的区域。新占据面必须从
        // 这里外推，不能因为新高度已 active 就错误保留上一时刻的干区零速度。
        boundary.BuildFaceMask(scratch.sandActive, scratch.sandFaceValid);
        boundary.IntersectFaceMasks(
            scratch.sandFaceValid,
            scratch.sandTerrainFaceOpen,
            scratch.sandFaceValid);
        scratch.oldSandHeight = state.sand.height;

        // 必须在 AdvectHeight 覆盖 h 之前构造 h*theta 和 h*P。
        ParallelForEach(state.sand.height.GetGrid(), [&](Vector2i const &cell)
        {
            scratch.scalar1[cell] =
                scratch.oldSandHeight[cell] *
                state.sand.rotationAngle[cell];
            scratch.matrix0[cell] =
                scratch.oldSandHeight[cell] *
                state.sand.stretch[cell];
        });
        AdvectionFV::SolveConservedWithCarrier(
            scratch.scalar1,
            scratch.oldSandHeight,
            state.sand.velocity,
            dt,
            boundary,
            scratch.saturation);
        AdvectionFV::SolveConservedWithCarrier(
            scratch.matrix0,
            scratch.oldSandHeight,
            state.sand.velocity,
            dt,
            boundary,
            scratch.matrix1);

        if (config.physics.saturationMode ==
            SaturationEvolutionMode::FrozenBound)
        {
            // 诊断模式下 mixed water 是附着在沙质量上的守恒示踪量。使用与
            // h_s 完全相同的 carrier 和 donor limiter，可让局部 h_m/h_s 随沙
            // 输运，而不是被独立水流、扩散或容量转换重新分布。
            AdvectionFV::SolveConservedWithCarrier(
                state.mixedWater.height,
                scratch.oldSandHeight,
                state.sand.velocity,
                dt,
                boundary,
                scratch.totalWater);
        }

        AdvectionStats const heightStats =
            AdvectHeight(state.sand, dt, boundary, scratch);
        if (config.physics.saturationMode ==
            SaturationEvolutionMode::FrozenBound)
        {
            state.mixedWater.height = scratch.totalWater;
            boundary.ApplyHeightBoundary(state.mixedWater.height);
            // FrozenBound 不允许生成独立自由水；保留显式清零可防止旧缓存或
            // 后续重构意外把动态水层重新带入该消融实验。
            state.pureWater.SetZero();
        }
        report.cumulativeAdvectionVolumeChange += heightStats.VolumeChange();
        AdvectConstitutiveState(
            state.sand,
            scratch.oldSandHeight,
            state.sand.velocity,
            dt,
            boundary,
            config,
            scratch);

        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        if (config.physics.projectDeformationAfterAdvection)
        {
            // h、theta 和 P 已经到达目的单元，且新沙高可能改变了这里的
            // phi=water/(water+sand)。必须让返回映射和紧随其后的应力计算
            // 读取同一份目的单元含水状态；子步开头的投影不能替代这一步。
            ProjectionStats const projection =
                ProjectDeformationGradient(
                    state, config, scratch, true);
            report.maxPostAdvectionYieldResidualBefore = std::max(
                report.maxPostAdvectionYieldResidualBefore,
                projection.maximumYieldResidualBefore);
            report.maxPostAdvectionYieldResidualAfter = std::max(
                report.maxPostAdvectionYieldResidualAfter,
                projection.maximumYieldResidualAfter);
            report.maxPostAdvectionProjectionDelta = std::max(
                report.maxPostAdvectionProjectionDelta,
                projection.maximumDeformationDelta);
            report.postAdvectionProjectionCount +=
                projection.changedCellCount;
        }
        else
        {
            // 开关关闭时显式清空，防止复用 scratch 的调用方把上一步诊断
            // 误认为本步执行过平流后投影。
            scratch.postAdvectionYieldResidualBefore.SetZero();
            scratch.postAdvectionYieldResidualAfter.SetZero();
            scratch.postAdvectionProjectionDelta.SetZero();
        }
        boundary.BuildSandTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            config.physics.sandTerrainContactTolerance,
            scratch.sandTerrainFaceOpen);
        boundary.BuildFaceMask(scratch.sandActive, scratch.faceValid);
        boundary.IntersectFaceMasks(
            scratch.faceValid,
            scratch.sandTerrainFaceOpen,
            scratch.faceValid);
        AdvectVelocity(
            state.sand,
            scratch.sandFaceValid,
            dt,
            boundary,
            scratch);
        boundary.ApplyTerrainContactBoundary(
            state.sand.velocity, scratch.sandTerrainFaceOpen);

        BuildSandPotential(state, config, boundary, scratch);
        report.maxGravityAcceleration = std::max(
            report.maxGravityAcceleration,
            ApplyGravityAndWaterLoad(
                state.sand,
                scratch.sandPotential,
                config.physics.gravity,
                dt,
                boundary,
                scratch.faceValid));

        // 论文把阻力放在 Integrate_Sand_Velocity 的末尾：必须读取已经平流
        // 且施加重力后的沙速度，同时发生在应力内力和水层容量修正之前。
        if (config.physics.saturationMode == SaturationEvolutionMode::Dynamic)
        {
            ExchangeMomentumWithMixedWater(
                state,
                config,
                dt,
                boundary,
                scratch.faceValid,
                scratch,
                report);
        }
        else
        {
            // 束缚水与沙不存在独立相对速度，因此没有 M5 交换冲量。
            scratch.sandWaterImpulse.SetZero();
        }

        ComputeStress(state, config, scratch);
        ApplyInternalForce(
            state,
            config,
            dt,
            boundary,
            scratch,
            report);

        // 摩擦必须等到 Simulation 用新沙高修正 pure/mixed 水层后才执行；
        // 这里仅清理当前动力学阶段的无效面，防止临时速度污染层转换。
        FinalizeVelocity(
            state.sand,
            config.time.maxVelocity,
            boundary,
            scratch.faceValid,
            report);
        report.volumeAfterStep = ComputeVolume(state.sand.height, boundary);
        report.minHeight = std::min(
            report.minHeight,
            ComputeMinHeight(state.sand.height, boundary));
    }

    void SandSolver::ApplyFrictionAndFinalize(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions &boundary,
        SimulationScratch &scratch,
        SandStepReport &report) const
    {
        if (!(dt > 0.0) || !std::isfinite(dt))
        {
            throw std::invalid_argument(
                "sand friction time step must be finite and positive");
        }

        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        boundary.BuildSandTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            config.physics.sandTerrainContactTolerance,
            scratch.sandTerrainFaceOpen);
        boundary.BuildFaceMask(scratch.sandActive, scratch.faceValid);
        boundary.IntersectFaceMasks(
            scratch.faceValid,
            scratch.sandTerrainFaceOpen,
            scratch.faceValid);
        // 本阶段使用最新沙高重新判断接触：层容量修正可能改变水载荷，但不会
        // 改变几何判据 H+hs。组合掩码确保摩擦、最终速度和 F 更新观察一致边界。
        ApplyFriction(
            state,
            config,
            dt,
            boundary,
            scratch.faceValid,
            scratch,
            report);
        // M6.6：F 必须读取摩擦和速度保护之后真正提交到状态的速度。旧顺序
        // 会先用重力/内力产生的候选速度改变 F，再由静摩擦把速度清零，导致
        // “外形静止但 det(F) 逐步漂移”的算子分裂蠕变。
        FinalizeVelocity(
            state.sand,
            config.time.maxVelocity,
            boundary,
            scratch.faceValid,
            report);
        EvolveDeformationGradient(
            state,
            config,
            dt,
            scratch);
        // 返回映射仍在每个子步结束时执行，保证输出帧和下一子步都只观察
        // 合法弹性状态；子步开头的投影则保留为输入状态防御检查。
        ProjectDeformationGradient(state, config, scratch);
        boundary.ApplyHeightBoundary(state.sand.height);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);

        report.volumeAfterStep = ComputeVolume(state.sand.height, boundary);
        report.minHeight = std::min(
            report.minHeight,
            ComputeMinHeight(state.sand.height, boundary));
    }

    // ----- conservative sand-height advection ----------------------------------

    AdvectionStats SandSolver::AdvectHeight(
        SandState &sand,
        double dt,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        AdvectionStats const stats = AdvectionFV::SolveHeight(
            sand.height,
            sand.velocity,
            dt,
            boundary,
            scratch.scalar0);
        sand.height = scratch.scalar0;
        boundary.ApplyHeightBoundary(sand.height);
        return stats;
    }

    // ----- semi-Lagrangian sand-velocity advection ------------------------------

    void SandSolver::AdvectVelocity(
        SandState &sand,
        SGridData<std::uint8_t> const &sourceValidFaces,
        double dt,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        scratch.velocity0 = sand.velocity;
        Extrapolation::Solve(
            scratch.velocity0,
            0.0,
            2,
            [&](int axis, Vector2i const &face)
            {
                return sourceValidFaces[axis][face] != 0;
            });
        boundary.ApplyVelocityBoundary(scratch.velocity0);

        AdvectionSL::Solve(
            scratch.velocity0,
            scratch.velocity0,
            dt,
            boundary,
            sand.velocity);
        boundary.ApplyVelocityBoundary(sand.velocity);
    }

    // ----- M4 polar decomposition, transport, and plastic projection ------------

    Matrix2d SandSolver::ReconstructDeformationGradient(
        double rotationAngle,
        Matrix2d const &stretch)
    {
        double const cosine = std::cos(rotationAngle);
        double const sine = std::sin(rotationAngle);
        Matrix2d rotation;
        rotation << cosine, -sine, sine, cosine;
        return rotation * stretch;
    }

    SandSolver::ProjectionStats SandSolver::ProjectDeformationGradient(
        SimulationState &state,
        SimulationConfig const &config,
        SimulationScratch &scratch,
        bool capturePostAdvectionDiagnostics) const
    {
        ProjectionStats stats;
        SandConstitutive const constitutive(
            config.physics,
            config.time.determinantEpsilon);
        // 返回映射和随后的应力计算必须使用同一种饱和度定义。这里按当前状态
        // 重新构造，避免 scratch.saturation 曾被 h*theta 平流临时复用而污染。
        BuildConstitutiveSaturation(state, config, scratch);

        if (capturePostAdvectionDiagnostics)
        {
            scratch.postAdvectionYieldResidualBefore.SetZero();
            scratch.postAdvectionYieldResidualAfter.SetZero();
            scratch.postAdvectionProjectionDelta.SetZero();
        }

        // active F 先极分解，并用该单元上一步的 theta 选择最接近的 2pi 分支，
        // 使持续旋转跨过 +/-pi 时不会产生守恒量跳跃。
        ParallelForEach(state.sand.height.GetGrid(), [&](Vector2i const &cell)
        {
            if (!scratch.sandActive[cell])
            {
                return;
            }
            double rawAngle = 0.0;
            Matrix2d stretch;
            constitutive.PolarDecompose(
                state.sand.deformationGradient[cell],
                rawAngle,
                stretch);
            double const oldAngle = state.sand.rotationAngle[cell];
            state.sand.rotationAngle[cell] =
                rawAngle + 2.0 * std::numbers::pi *
                    std::round(
                        (oldAngle - rawAngle) /
                        (2.0 * std::numbers::pi));
            state.sand.stretch[cell] = stretch;
        });

        // 对每个连通沙区做空间解缠，避免相邻 +179/-179 度在 h*theta
        // 平流时抵消。这里故意串行：连通分量传播有数据依赖且只占 M4 小部分。
        Grid const &grid = state.sand.height.GetGrid();
        std::vector<std::uint8_t> &visited =
            scratch.constitutiveVisited;
        std::fill(visited.begin(), visited.end(), std::uint8_t{0});
        std::vector<Vector2i> &frontier =
            scratch.constitutiveFrontier;
        ForEach(grid, [&](Vector2i const &seed)
        {
            int const seedIndex = grid.IndexOf(seed);
            if (!scratch.sandActive[seed] || visited[seedIndex])
            {
                return;
            }
            frontier.clear();
            frontier.push_back(seed);
            visited[seedIndex] = 1;
            for (std::size_t head = 0; head < frontier.size(); ++head)
            {
                Vector2i const cell = frontier[head];
                double const reference =
                    state.sand.rotationAngle[cell];
                for (int ordinal = 0;
                     ordinal < Grid::GetNumNeighbors();
                     ++ordinal)
                {
                    Vector2i const neighbor =
                        Grid::NeighborOf(cell, ordinal);
                    if (!grid.IsValid(neighbor) ||
                        !scratch.sandActive[neighbor])
                    {
                        continue;
                    }
                    int const index = grid.IndexOf(neighbor);
                    if (visited[index])
                    {
                        continue;
                    }
                    state.sand.rotationAngle[neighbor] =
                        reference + std::remainder(
                            state.sand.rotationAngle[neighbor] - reference,
                            2.0 * std::numbers::pi);
                    visited[index] = 1;
                    frontier.push_back(neighbor);
                }
            }
        });

        Extrapolation::ExtrapolateRotationStretchOneRing3x3(
            state.sand.rotationAngle,
            state.sand.stretch,
            scratch.sandActive);

        ParallelForEach(state.sand.height.GetGrid(), [&](Vector2i const &cell)
        {
            Matrix2d reconstructed =
                ReconstructDeformationGradient(
                    state.sand.rotationAngle[cell],
                    state.sand.stretch[cell]);
            if (scratch.sandActive[cell])
            {
                double const saturation = scratch.saturation[cell];
                bool projectionChanged = false;
                Matrix2d const trial = reconstructed;
                if (capturePostAdvectionDiagnostics)
                {
                    Matrix2d const trialStress =
                        constitutive.ComputeCauchyStress(
                            trial, saturation);
                    scratch.postAdvectionYieldResidualBefore[cell] =
                        constitutive.YieldResidual(
                            trialStress, saturation);
                }
                reconstructed =
                    constitutive.Project(
                        reconstructed,
                        saturation,
                        projectionChanged);

                if (projectionChanged)
                {
                    // 只有返回映射真正改变 P 时才需再次极分解。弹性状态的
                    // theta/P 正是上面重建 F 的输入，重复 SVD 不提供新信息。
                    double projectedAngle = 0.0;
                    Matrix2d projectedStretch;
                    constitutive.PolarDecompose(
                        reconstructed,
                        projectedAngle,
                        projectedStretch);
                    projectedAngle +=
                        2.0 * std::numbers::pi *
                        std::round(
                            (state.sand.rotationAngle[cell] - projectedAngle) /
                            (2.0 * std::numbers::pi));
                    state.sand.rotationAngle[cell] = projectedAngle;
                    state.sand.stretch[cell] = projectedStretch;
                }
                if (capturePostAdvectionDiagnostics)
                {
                    Matrix2d const projectedStress =
                        constitutive.ComputeCauchyStress(
                            reconstructed, saturation);
                    scratch.postAdvectionYieldResidualAfter[cell] =
                        constitutive.YieldResidual(
                            projectedStress, saturation);
                    scratch.postAdvectionProjectionDelta[cell] =
                        (reconstructed - trial).norm();
                }
            }
            state.sand.deformationGradient[cell] = reconstructed;
        });

        if (capturePostAdvectionDiagnostics)
        {
            // 统计只遍历已经写完的逐单元场，避免在 ParallelForEach 内对共享
            // max/count 做原子更新。count 是跨沙子子步累计的“修改次数”，
            // 不是一帧内去重后的网格数量。
            ForEach(state.sand.height.GetGrid(), [&](Vector2i const &cell)
            {
                if (!scratch.sandActive[cell])
                {
                    return;
                }
                stats.maximumYieldResidualBefore = std::max(
                    stats.maximumYieldResidualBefore,
                    scratch.postAdvectionYieldResidualBefore[cell]);
                stats.maximumYieldResidualAfter = std::max(
                    stats.maximumYieldResidualAfter,
                    scratch.postAdvectionYieldResidualAfter[cell]);
                double const delta =
                    scratch.postAdvectionProjectionDelta[cell];
                stats.maximumDeformationDelta = std::max(
                    stats.maximumDeformationDelta, delta);
                if (delta > 64.0 * std::numeric_limits<double>::epsilon())
                {
                    ++stats.changedCellCount;
                }
            });
        }
        return stats;
    }

    void SandSolver::BuildConstitutiveSaturation(
        SimulationState const &state,
        SimulationConfig const &config,
        SimulationScratch &scratch) const
    {
        Grid const &grid = state.sand.height.GetGrid();
        ParallelForEach(grid, [&](Vector2i const &cell)
        {
            if (!scratch.sandActive[cell])
            {
                scratch.rawConstitutiveSaturation[cell] = 0.0;
                return;
            }
            double const water =
                state.pureWater.height[cell] +
                state.mixedWater.height[cell];
            double const total = water + state.sand.height[cell];
            scratch.rawConstitutiveSaturation[cell] =
                total > config.time.heightEpsilon
                ? std::clamp(water / total, 0.0, 1.0)
                : 0.0;
        });

        if (!config.physics.constitutiveSaturationLowpass)
        {
            scratch.saturation = scratch.rawConstitutiveSaturation;
            return;
        }

        // 一次可分离的 [1,2,1]^T [1,2,1] binomial 核，相当于一格尺度的
        // 温和低通。只纳入有效沙单元并重新归一化，避免边界外的零值把薄沙
        // 人为变干；因而本消融测量的是沙体内部本构输入的空间平滑作用。
        ParallelForEach(grid, [&](Vector2i const &cell)
        {
            if (!scratch.sandActive[cell])
            {
                scratch.saturation[cell] = 0.0;
                return;
            }
            double weightedSum = 0.0;
            double weightSum = 0.0;
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
                    weightedSum +=
                        weight * scratch.rawConstitutiveSaturation[neighbor];
                    weightSum += weight;
                }
            }
            scratch.saturation[cell] =
                weightSum > 0.0
                ? weightedSum / weightSum
                : scratch.rawConstitutiveSaturation[cell];
        });
    }

    void SandSolver::AdvectConstitutiveState(
        SandState &sand,
        GridData<double> const &,
        SGridData<double> const &,
        double,
        BoundaryConditions const &,
        SimulationConfig const &config,
        SimulationScratch &scratch) const
    {
        ParallelForEach(sand.height.GetGrid(), [&](Vector2i const &cell)
        {
            double const height = sand.height[cell];
            if (height <= config.time.heightEpsilon)
            {
                sand.rotationAngle[cell] = 0.0;
                sand.stretch[cell] = Matrix2d::Identity();
                sand.deformationGradient[cell] = Matrix2d::Identity();
                return;
            }

            sand.rotationAngle[cell] =
                scratch.saturation[cell] / height;
            Matrix2d transported =
                scratch.matrix1[cell] / height;
            transported =
                0.5 * (transported + transported.transpose());

            // 正权重迎风平流理论上保持 P 正定；特征值下界只清理舍入误差，
            // 并在极薄新激活单元中避免奇异 F。
            SelfAdjointEigenSolver<Matrix2d> const eigen(transported);
            if (eigen.info() != Success)
            {
                throw std::runtime_error(
                    "transported sand stretch eigensolve failed");
            }
            Vector2d const values =
                eigen.eigenvalues().cwiseMax(
                    std::sqrt(config.time.determinantEpsilon));
            sand.stretch[cell] =
                eigen.eigenvectors() * values.asDiagonal() *
                eigen.eigenvectors().transpose();
            sand.deformationGradient[cell] =
                ReconstructDeformationGradient(
                    sand.rotationAngle[cell],
                    sand.stretch[cell]);
        });
    }

    void SandSolver::ComputeStress(
        SimulationState const &state,
        SimulationConfig const &config,
        SimulationScratch &scratch) const
    {
        SandConstitutive const constitutive(
            config.physics,
            config.time.determinantEpsilon);
        BuildConstitutiveSaturation(state, config, scratch);
        ParallelForEach(state.sand.height.GetGrid(), [&](Vector2i const &cell)
        {
            double const sandHeight =
                state.sand.height[cell];
            if (!scratch.sandActive[cell])
            {
                scratch.saturation[cell] = 0.0;
                scratch.stress[cell] = Matrix2d::Zero();
                scratch.weightedStress[cell] = Matrix2d::Zero();
                scratch.internalYieldResidual[cell] = 0.0;
                return;
            }
            scratch.stress[cell] =
                constitutive.ComputeCauchyStress(
                    state.sand.deformationGradient[cell],
                    scratch.saturation[cell]);
            // 保存内力阶段的真实 trial 状态。帧末 F 已经经过速度更新和返回
            // 映射，重算的 yield_residual 无法回答本次内力是否读到了越界 F。
            scratch.internalYieldResidual[cell] =
                constitutive.YieldResidual(
                    scratch.stress[cell], scratch.saturation[cell]);
            scratch.weightedStress[cell] =
                sandHeight * scratch.stress[cell];
        });
    }

    SandSolver::InternalAccelerationStats
    SandSolver::ComputeInternalAcceleration(
        GridData<double> const &sandHeight,
        GridData<Matrix2d> const &weightedStress,
        double sandDensity,
        double heightEpsilon,
        double heightFloor,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &validFaces,
        SGridData<double> &acceleration,
        SGridData<double> &normalAccelerationField,
        SGridData<double> &tangentialAccelerationField,
        SGridData<double> &faceHeightField,
        SGridData<double> &effectiveHeightField) const
    {
        InternalAccelerationStats stats;
        double const invDx =
            sandHeight.GetGrid().GetInvSpacing();
        ForEach(acceleration.GetGrids(), [&](int axis, Vector2i const &face)
        {
            if (!validFaces[axis][face] ||
                !boundary.IsInsideFace(axis, face) ||
                boundary.IsNormalBoundaryFace(axis, face))
            {
                acceleration[axis][face] = 0.0;
                normalAccelerationField[axis][face] = 0.0;
                tangentialAccelerationField[axis][face] = 0.0;
                faceHeightField[axis][face] = 0.0;
                effectiveHeightField[axis][face] = 0.0;
                return;
            }

            Vector2i const left =
                StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const right =
                StaggeredGrid::AdjCellOfFace(axis, face, 1);
            int const tangent = axis ^ 1;
            Vector2i const tangentOffset =
                Vector2i::Unit(tangent);

            // 法向分量直接跨面差分；剪切分量先平均到目标面，再沿切向
            // 作中心差分。ghost 区 Q=0 等价于首版采用的零牵引边界。
            double const normalDerivative =
                (weightedStress[right](axis, axis) -
                 weightedStress[left](axis, axis)) *
                invDx;
            double const tangentUpper = 0.5 * (
                weightedStress.At(left + tangentOffset)(axis, tangent) +
                weightedStress.At(right + tangentOffset)(axis, tangent));
            double const tangentLower = 0.5 * (
                weightedStress.At(left - tangentOffset)(axis, tangent) +
                weightedStress.At(right - tangentOffset)(axis, tangent));
            double const tangentDerivative =
                0.5 * invDx *
                (tangentUpper - tangentLower);
            double const faceHeight =
                0.5 * (
                    std::max(sandHeight[left], 0.0) +
                    std::max(sandHeight[right], 0.0));
            faceHeightField[axis][face] = faceHeight;
            if (faceHeight <= heightEpsilon)
            {
                acceleration[axis][face] = 0.0;
                normalAccelerationField[axis][face] = 0.0;
                tangentialAccelerationField[axis][face] = 0.0;
                effectiveHeightField[axis][face] = 0.0;
                return;
            }
            // heightFloor 只正则化惯性分母，不改变 h*sigma 的守恒散度。
            // 这等价于给自由表面内力设置一层最小有效柱质量。
            double const effectiveHeight =
                std::max({faceHeight, heightFloor, heightEpsilon});
            double const normalAcceleration =
                normalDerivative / (sandDensity * effectiveHeight);
            double const tangentialAcceleration =
                tangentDerivative / (sandDensity * effectiveHeight);
            normalAccelerationField[axis][face] = normalAcceleration;
            tangentialAccelerationField[axis][face] = tangentialAcceleration;
            effectiveHeightField[axis][face] = effectiveHeight;
            acceleration[axis][face] =
                normalAcceleration + tangentialAcceleration;
            stats.maximumNormal = std::max(
                stats.maximumNormal, std::abs(normalAcceleration));
            stats.maximumTangential = std::max(
                stats.maximumTangential, std::abs(tangentialAcceleration));
            if (std::abs(acceleration[axis][face]) > stats.maximum)
            {
                stats.maximum = std::abs(acceleration[axis][face]);
                stats.heightAtMaximum = faceHeight;
            }
        });
        return stats;
    }

    void SandSolver::ApplyInternalForce(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch,
        SandStepReport &report) const
    {
        if (!config.physics.enableSandInternalForce)
        {
            scratch.internalAcceleration.SetZero();
            scratch.internalNormalAcceleration.SetZero();
            scratch.internalTangentialAcceleration.SetZero();
            scratch.internalFaceHeight.SetZero();
            scratch.internalEffectiveHeight.SetZero();
        }
        InternalAccelerationStats const stats =
            config.physics.enableSandInternalForce
            ? ComputeInternalAcceleration(
                state.sand.height,
                scratch.weightedStress,
                config.physics.sandDensity,
                config.time.heightEpsilon,
                std::max(
                    config.time.internalForceHeightFloor,
                    config.time.internalForceHeightFraction *
                        state.sand.height.GetGrid().GetSpacing()),
                boundary,
                scratch.faceValid,
                scratch.internalAcceleration,
                scratch.internalNormalAcceleration,
                scratch.internalTangentialAcceleration,
                scratch.internalFaceHeight,
                scratch.internalEffectiveHeight)
            : InternalAccelerationStats{};
        report.maxInternalAcceleration = std::max(
            report.maxInternalAcceleration,
            stats.maximum);
        report.maxNormalStressAcceleration = std::max(
            report.maxNormalStressAcceleration,
            stats.maximumNormal);
        report.maxTangentialStressAcceleration = std::max(
            report.maxTangentialStressAcceleration,
            stats.maximumTangential);
        if (stats.maximum >= report.maxInternalAcceleration)
        {
            report.heightAtMaxInternalAcceleration =
                stats.heightAtMaximum;
        }

        ParallelForEach(state.sand.velocity.GetGrids(),
            [&](int axis, Vector2i const &face)
        {
            if (scratch.faceValid[axis][face] &&
                boundary.IsInsideFace(axis, face))
            {
                state.sand.velocity[axis][face] +=
                    dt * scratch.internalAcceleration[axis][face];
            }
        });
        boundary.ApplyVelocityBoundary(state.sand.velocity);
    }

    void SandSolver::EvolveDeformationGradient(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        SimulationScratch const &scratch) const
    {
        ParallelForEach(state.sand.height.GetGrid(), [&](Vector2i const &cell)
        {
            if (!scratch.sandActive[cell])
            {
                return;
            }
            Matrix2d const velocityGradient =
                MacOperators::VelocityGradientAtCell(
                    state.sand.velocity, cell);
            Matrix2d const trial =
                state.sand.deformationGradient[cell] +
                dt * velocityGradient *
                    state.sand.deformationGradient[cell];
            if (!trial.array().isFinite().all() ||
                trial.determinant() <=
                    config.time.determinantEpsilon)
            {
                throw std::runtime_error(fmt::format(
                    "sand kinematic update inverted F at ({}, {})",
                    cell.x(), cell.y()));
            }
            state.sand.deformationGradient[cell] = trial;
        });
    }

    // ----- sand-column gravity and pure-water load ------------------------------

    void SandSolver::BuildSandPotential(
        SimulationState const &state,
        SimulationConfig const &config,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        double const waterToSandDensity =
            config.physics.waterDensity / config.physics.sandDensity;
        ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
        {
            scratch.sandPotential[cell] =
                state.terrain[cell] +
                state.sand.height[cell] +
                waterToSandDensity * state.pureWater.height[cell];
        });
        boundary.ApplyHeightBoundary(scratch.sandPotential);
    }

    double SandSolver::ApplyGravityAndWaterLoad(
        SandState &sand,
        GridData<double> const &potential,
        double gravity,
        double dt,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &validFaces) const
    {
        double maximumAcceleration = 0.0;
        ForEach(sand.velocity.GetGrids(), [&](int axis, Vector2i const &face)
        {
            if (!validFaces[axis][face] ||
                !boundary.IsInsideFace(axis, face) ||
                boundary.IsNormalBoundaryFace(axis, face))
            {
                sand.velocity[axis][face] = 0.0;
                return;
            }

            double const acceleration =
                -gravity * MacOperators::GradientAtFace(
                    potential, axis, face);
            sand.velocity[axis][face] += dt * acceleration;
            maximumAcceleration =
                std::max(maximumAcceleration, std::abs(acceleration));
        });
        return maximumAcceleration;
    }

    // ----- M5 sand--mixed-water momentum exchange -------------------------------

    void SandSolver::ExchangeMomentumWithMixedWater(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &sandValidFaces,
        SimulationScratch &scratch,
        SandStepReport &report) const
    {
        // sandWaterImpulse[axis][face] 的正号表示该分量的动量从沙柱流向
        // mixed-water 柱。即使本步关闭交换也清零，避免诊断快照误读旧缓存。
        scratch.sandWaterImpulse.SetZero();
        double const exchangeRate =
            config.physics.momentumExchangeRate;
        if (exchangeRate == 0.0)
        {
            return;
        }

        double const sandDensity = config.physics.sandDensity;
        double const waterDensity = config.physics.waterDensity;
        double const heightEpsilon = config.time.heightEpsilon;

        // 这里串行遍历是有意的：每个面仍独立更新，但报告中的 max/count
        // 需要确定性归约，且交换阶段远小于平流与本构计算。
        ForEach(state.sand.velocity.GetGrids(),
            [&](int axis, Vector2i const &face)
        {
            if (!sandValidFaces[axis][face] ||
                !boundary.IsInsideFace(axis, face) ||
                boundary.IsNormalBoundaryFace(axis, face))
            {
                return;
            }

            // 面柱高使用相邻两个单元的算术平均，与重力、应力和摩擦模块
            // 对 MAC 面质量的定义保持一致。
            double const sandHeight = std::max(
                MacOperators::AverageToFace(
                    state.sand.height, axis, face),
                0.0);
            double const mixedHeight = std::max(
                MacOperators::AverageToFace(
                    state.mixedWater.height, axis, face),
                0.0);
            if (sandHeight <= heightEpsilon ||
                mixedHeight <= heightEpsilon)
            {
                return;
            }

            // 质量均按单位水平面积计。阻力系数 K=c_e*h_m，使沙加速度
            // K(v_s-v_m)/(rho_s*h_s) 与论文式 4.2.3 完全对应。
            double const sandColumnMass =
                sandDensity * sandHeight;
            double const mixedColumnMass =
                waterDensity * mixedHeight;
            double const dragCoefficient =
                exchangeRate * mixedHeight;
            double const sandVelocity =
                state.sand.velocity[axis][face];
            double const mixedVelocity =
                state.mixedWater.velocity[axis][face];
            double const relativeBefore =
                sandVelocity - mixedVelocity;

            double impulse = 0.0;
            if (config.exchangeScheme ==
                SandWaterExchangeScheme::PaperSequential)
            {
                // 显式论文顺序：用旧相对速度计算沙加速度，再把完全相反的
                // 冲量加入混合水。它守恒动量，但大 dt*c_e 时可能越过平衡点。
                impulse =
                    dt * dragCoefficient * relativeBefore;
            }
            else
            {
                // 对 d'=-(K/ms+K/mm)d 作后向 Euler 的解析解。
                // J=dt*K*d^{n+1} 是同一阻力在整个时间步的等效冲量。
                double const dampingDenominator =
                    1.0 + dt * dragCoefficient *
                        (1.0 / sandColumnMass +
                         1.0 / mixedColumnMass);
                double const relativeAfter =
                    relativeBefore / dampingDenominator;
                impulse =
                    dt * dragCoefficient * relativeAfter;
            }

            double const momentumBefore =
                sandColumnMass * sandVelocity +
                mixedColumnMass * mixedVelocity;
            state.sand.velocity[axis][face] =
                sandVelocity - impulse / sandColumnMass;
            state.mixedWater.velocity[axis][face] =
                mixedVelocity + impulse / mixedColumnMass;
            scratch.sandWaterImpulse[axis][face] = impulse;

            double const relativeAfter =
                state.sand.velocity[axis][face] -
                state.mixedWater.velocity[axis][face];
            double const momentumAfter =
                sandColumnMass *
                    state.sand.velocity[axis][face] +
                mixedColumnMass *
                    state.mixedWater.velocity[axis][face];
            report.maxExchangeRelativeSpeedBefore = std::max(
                report.maxExchangeRelativeSpeedBefore,
                std::abs(relativeBefore));
            report.maxExchangeRelativeSpeedAfter = std::max(
                report.maxExchangeRelativeSpeedAfter,
                std::abs(relativeAfter));
            report.maxExchangeImpulse = std::max(
                report.maxExchangeImpulse,
                std::abs(impulse));
            report.maxExchangeMomentumError = std::max(
                report.maxExchangeMomentumError,
                std::abs(momentumAfter - momentumBefore));
            ++report.exchangedFaceCount;
            if (relativeBefore * relativeAfter < 0.0)
            {
                ++report.exchangeOvershootFaceCount;
            }
        });

        // 正常边界面从未交换，仍统一重施边界以清除切向 ghost 值。
        boundary.ApplyVelocityBoundary(state.sand.velocity);
        boundary.ApplyVelocityBoundary(state.mixedWater.velocity);
    }

    // ----- external Coulomb friction --------------------------------------------

    void SandSolver::ApplyFriction(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &validFaces,
        SimulationScratch &scratch,
        SandStepReport &report) const
    {
        // 两个分量必须都从同一份摩擦前快照读取。若先改 u 再重建 v 面速度，
        // 缩放因子会依赖循环顺序，并会人为旋转速度方向。
        scratch.velocity0 = state.sand.velocity;
        double const densitySand = config.physics.sandDensity;
        double const densityWater = config.physics.waterDensity;
        double const heightEpsilon = config.time.heightEpsilon;

        ForEach(state.sand.velocity.GetGrids(), [&](int axis, Vector2i const &face)
        {
            if (!validFaces[axis][face] ||
                !boundary.IsInsideFace(axis, face) ||
                boundary.IsNormalBoundaryFace(axis, face))
            {
                state.sand.velocity[axis][face] = 0.0;
                return;
            }

            double const sandHeight =
                MacOperators::AverageToFace(state.sand.height, axis, face);
            if (sandHeight <= heightEpsilon)
            {
                state.sand.velocity[axis][face] = 0.0;
                return;
            }
            double const mixedHeight = std::max(
                MacOperators::AverageToFace(
                    state.mixedWater.height, axis, face),
                0.0);

            // 浮力后的有效法向柱质量。max 防止过量混合水把摩擦变成加速。
            double const effectiveNormalMass = std::max(
                densitySand * sandHeight -
                    densityWater * mixedHeight,
                0.0);
            double const deceleration =
                config.physics.groundFriction *
                config.physics.gravity *
                effectiveNormalMass /
                (densitySand * sandHeight);
            report.maxFrictionDeceleration = std::max(
                report.maxFrictionDeceleration, deceleration);

            Vector2d const velocityAtFace =
                MacOperators::VelocityAtFace(
                    scratch.velocity0, axis, face);
            double const speed = velocityAtFace.norm();
            if (speed <= heightEpsilon)
            {
                state.sand.velocity[axis][face] = 0.0;
                return;
            }

            double const remainingSpeed =
                std::max(speed - dt * deceleration, 0.0);
            double const scale = remainingSpeed / speed;
            state.sand.velocity[axis][face] =
                scratch.velocity0[axis][face] * scale;
            if (remainingSpeed == 0.0)
            {
                ++report.frictionStoppedFaces;
            }
        });
    }

    void SandSolver::FinalizeVelocity(
        SandState &sand,
        double maxVelocity,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &validFaces,
        SandStepReport &report) const
    {
        // 统计和计数需要确定性归约；这一小段串行遍历也避免原子操作污染热路径。
        ForEach(sand.velocity.GetGrids(), [&](int axis, Vector2i const &face)
        {
            if (!validFaces[axis][face] ||
                !boundary.IsInsideFace(axis, face))
            {
                sand.velocity[axis][face] = 0.0;
                return;
            }
            double const magnitude =
                std::abs(sand.velocity[axis][face]);
            report.maxVelocityBeforeClamping = std::max(
                report.maxVelocityBeforeClamping, magnitude);
            if (magnitude > maxVelocity)
            {
                ++report.velocityClampCount;
            }
            sand.velocity[axis][face] = std::clamp(
                sand.velocity[axis][face],
                -maxVelocity,
                maxVelocity);
        });
        boundary.ApplyVelocityBoundary(sand.velocity);
    }

    double SandSolver::ComputeVolume(
        GridData<double> const &height,
        BoundaryConditions const &boundary)
    {
        double const area =
            height.GetGrid().GetSpacing() *
            height.GetGrid().GetSpacing();
        double volume = 0.0;
        ForEach(height.GetGrid(), [&](Vector2i const &cell)
        {
            if (boundary.IsInsideCell(cell))
            {
                volume += height[cell] * area;
            }
        });
        return volume;
    }

    double SandSolver::ComputeMinHeight(
        GridData<double> const &height,
        BoundaryConditions const &boundary)
    {
        double minimum = std::numeric_limits<double>::infinity();
        ForEach(height.GetGrid(), [&](Vector2i const &cell)
        {
            if (boundary.IsInsideCell(cell))
            {
                minimum = std::min(minimum, height[cell]);
            }
        });
        return std::isfinite(minimum) ? minimum : 0.0;
    }
}
