#include "physics/WaterSolver.h"

namespace Pivot
{
    namespace
    {
        bool HasActiveCells(GridData<std::uint8_t> const &mask)
        {
            return std::any_of(
                mask.GetData().begin(),
                mask.GetData().end(),
                [](std::uint8_t value)
                { return value != 0; });
        }

        /// 仅对拥有两个内部邻接单元的面做中心平均；封闭法向边界由调用者清零。
        double AverageInteriorCellsToFace(
            GridData<double> const &field,
            int axis,
            Vector2i const &face,
            BoundaryConditions const &boundary)
        {
            Vector2i const negative =
                StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const positive =
                StaggeredGrid::AdjCellOfFace(axis, face, 1);
            if (!boundary.IsInsideCell(negative) ||
                !boundary.IsInsideCell(positive))
            {
                return 0.0;
            }
            return 0.5 * (field[negative] + field[positive]);
        }
    }

    // ----- M2 step orchestration -------------------------------------------------

    WaterStepReport WaterSolver::Step(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions &boundary,
        SimulationScratch &scratch) const
    {
        if (!(dt > 0.0) || !std::isfinite(dt))
        {
            throw std::invalid_argument("water time step must be finite and positive");
        }

        boundary.ApplyHeightBoundary(state.pureWater.height);
        boundary.ApplyHeightBoundary(state.mixedWater.height);
        boundary.ApplyVelocityBoundary(state.pureWater.velocity);
        boundary.ApplyVelocityBoundary(state.mixedWater.velocity);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        boundary.BuildWaterTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            scratch.waterTerrainFaceOpen);
        // 高度平流发生在本步压力更新之前，因此必须先清除上一时刻可能残留
        // 在地形阻挡面上的速度；只在压力阶段 return 会允许质量先穿过高地。
        boundary.ApplyTerrainContactBoundary(
            state.pureWater.velocity, scratch.waterTerrainFaceOpen);
        boundary.ApplyTerrainContactBoundary(
            state.mixedWater.velocity, scratch.waterTerrainFaceOpen);

        WaterStepReport report;
        bool const hasPureWater =
            HasActiveCells(scratch.pureWaterActive);
        bool const hasMixedWater =
            HasActiveCells(scratch.mixedWaterActive);
        bool const hasWater = hasPureWater || hasMixedWater;
        if (!hasWater)
        {
            // M7 干场景快速路径：完整水步骤最终同样会把所有干面速度和转移
            // 冲量清零。直接提交该结果可跳过两层平流、压力和多次全场复制。
            state.pureWater.velocity.SetZero();
            state.mixedWater.velocity.SetZero();
            scratch.mixedHeightDelta.SetZero();
            scratch.waterTransferImpulse.SetZero();
            return report;
        }
        report.totalVolumeBefore = ComputeTotalWaterVolume(state, boundary);
        report.mixedVolumeBeforeDiffusion =
            ComputeVolume(state.mixedWater.height, boundary);

        // 1. 扩散只作用于沙内混合水；高度和速度均读取同一份旧时刻快照。
        if (hasMixedWater)
        {
            DiffuseMixedWater(
                state.mixedWater, config, dt, boundary, scratch);
        }
        report.mixedVolumeAfterDiffusion =
            ComputeVolume(state.mixedWater.height, boundary);
        report.totalVolumeAfterDiffusion =
            ComputeTotalWaterVolume(state, boundary);

        // 扩散可能改变 mixed-water 湿区，因此在高度平流前重建共享接触面，
        // 并再次清除扩散速度更新可能写入的阻挡面。
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        boundary.BuildWaterTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            scratch.waterTerrainFaceOpen);
        boundary.ApplyTerrainContactBoundary(
            state.pureWater.velocity, scratch.waterTerrainFaceOpen);
        boundary.ApplyTerrainContactBoundary(
            state.mixedWater.velocity, scratch.waterTerrainFaceOpen);

        // 保存 predictor 高度对应的速度有效区。它既要求该层邻接材料，也要求
        // 地形面开放；高度平流/层修正后新湿面仍从这份旧有效区外推。
        if (hasPureWater)
        {
            boundary.BuildFaceMask(
                scratch.pureWaterActive,
                scratch.pureWaterFaceValid);
            boundary.IntersectFaceMasks(
                scratch.pureWaterFaceValid,
                scratch.waterTerrainFaceOpen,
                scratch.pureWaterFaceValid);
        }
        else
        {
            scratch.pureWaterFaceValid.SetZero();
        }
        if (hasMixedWater)
        {
            boundary.BuildFaceMask(
                scratch.mixedWaterActive,
                scratch.mixedWaterFaceValid);
            boundary.IntersectFaceMasks(
                scratch.mixedWaterFaceValid,
                scratch.waterTerrainFaceOpen,
                scratch.mixedWaterFaceValid);
        }
        else
        {
            scratch.mixedWaterFaceValid.SetZero();
        }

        // 2. 两层高度分别使用各自速度做守恒迎风平流。
        if (hasPureWater)
        {
            AdvectHeight(state.pureWater, dt, boundary, scratch);
        }
        if (hasMixedWater)
        {
            AdvectHeight(state.mixedWater, dt, boundary, scratch);
        }
        report.totalVolumeAfterAdvection =
            ComputeTotalWaterVolume(state, boundary);

        // 3. 依据当前静态沙高恢复层容量约束，并记录 donor 方向。
        CorrectLayerHeightsAndBuildImpulse(
            state, config, dt, boundary, scratch, report);
        report.totalVolumeAfterLayerCorrection =
            ComputeTotalWaterVolume(state, boundary);

        // 高度改变后必须重建 wet/dry mask。两层的速度外推使用各自面掩码。
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        bool const hasPureWaterAfterCorrection =
            HasActiveCells(scratch.pureWaterActive);
        bool const hasMixedWaterAfterCorrection =
            HasActiveCells(scratch.mixedWaterActive);
        boundary.BuildWaterTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            scratch.waterTerrainFaceOpen);
        if (hasPureWaterAfterCorrection)
        {
            // AdvectVelocity 的掩码描述旧时刻可采样源区；当前目标区的材料与
            // 接触约束在随后 pressure/finalize 的组合掩码中统一提交。
            AdvectVelocity(
                state.pureWater,
                scratch.pureWaterFaceValid,
                dt,
                boundary,
                scratch);
            boundary.ApplyTerrainContactBoundary(
                state.pureWater.velocity, scratch.waterTerrainFaceOpen);
        }
        else
        {
            state.pureWater.velocity.SetZero();
        }
        if (hasMixedWaterAfterCorrection)
        {
            AdvectVelocity(
                state.mixedWater,
                scratch.mixedWaterFaceValid,
                dt,
                boundary,
                scratch);
            boundary.ApplyTerrainContactBoundary(
                state.mixedWater.velocity, scratch.waterTerrainFaceOpen);
        }
        else
        {
            state.mixedWater.velocity.SetZero();
        }

        // 4. 两层共享同一自由液面势，但保留彼此独立的速度。
        BuildHydraulicPotential(state, boundary, scratch);
        if (hasPureWaterAfterCorrection)
        {
            boundary.BuildFaceMask(
                scratch.pureWaterActive, scratch.faceValid);
            boundary.IntersectFaceMasks(
                scratch.faceValid,
                scratch.waterTerrainFaceOpen,
                scratch.faceValid);
            ApplyHydrostaticPressure(
                state.pureWater,
                scratch.hydraulicPotential,
                config.physics.gravity,
                dt,
                boundary,
                scratch.faceValid);
        }
        if (hasMixedWaterAfterCorrection)
        {
            boundary.BuildFaceMask(
                scratch.mixedWaterActive, scratch.faceValid);
            boundary.IntersectFaceMasks(
                scratch.faceValid,
                scratch.waterTerrainFaceOpen,
                scratch.faceValid);
            ApplyHydrostaticPressure(
                state.mixedWater,
                scratch.hydraulicPotential,
                config.physics.gravity,
                dt,
                boundary,
                scratch.faceValid);
        }

        // 5. predictor 中保存的层转换量在速度更新后按 donor 速度重组动量。
        report.maxLayerMomentumError =
            ApplyLayerTransferImpulse(
                state, config, boundary, scratch);

        if (hasPureWaterAfterCorrection)
        {
            boundary.BuildFaceMask(
                scratch.pureWaterActive, scratch.faceValid);
            boundary.IntersectFaceMasks(
                scratch.faceValid,
                scratch.waterTerrainFaceOpen,
                scratch.faceValid);
            FinalizeVelocity(
                state.pureWater,
                config.time.maxVelocity,
                boundary,
                scratch.faceValid);
        }
        if (hasMixedWaterAfterCorrection)
        {
            boundary.BuildFaceMask(
                scratch.mixedWaterActive, scratch.faceValid);
            boundary.IntersectFaceMasks(
                scratch.faceValid,
                scratch.waterTerrainFaceOpen,
                scratch.faceValid);
            FinalizeVelocity(
                state.mixedWater,
                config.time.maxVelocity,
                boundary,
                scratch.faceValid);
        }

        boundary.ApplyHeightBoundary(state.pureWater.height);
        boundary.ApplyHeightBoundary(state.mixedWater.height);
        boundary.ApplyVelocityBoundary(state.pureWater.velocity);
        boundary.ApplyVelocityBoundary(state.mixedWater.velocity);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);

        report.totalVolumeAfterStep = ComputeTotalWaterVolume(state, boundary);
        report.minHeight = ComputeMinWaterHeight(state, boundary);
        return report;
    }

    WaterStepReport WaterSolver::CorrectLayersAfterSandMotion(
        SimulationState &state,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions &boundary,
        SimulationScratch &scratch) const
    {
        if (!(dt > 0.0) || !std::isfinite(dt))
        {
            throw std::invalid_argument(
                "layer-correction time step must be finite and positive");
        }

        boundary.ApplyHeightBoundary(state.sand.height);
        boundary.ApplyHeightBoundary(state.pureWater.height);
        boundary.ApplyHeightBoundary(state.mixedWater.height);
        boundary.ApplyVelocityBoundary(state.pureWater.velocity);
        boundary.ApplyVelocityBoundary(state.mixedWater.velocity);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        boundary.BuildWaterTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            scratch.waterTerrainFaceOpen);
        boundary.ApplyTerrainContactBoundary(
            state.pureWater.velocity, scratch.waterTerrainFaceOpen);
        boundary.ApplyTerrainContactBoundary(
            state.mixedWater.velocity, scratch.waterTerrainFaceOpen);

        WaterStepReport report;
        bool const hasWater =
            HasActiveCells(scratch.pureWaterActive) ||
            HasActiveCells(scratch.mixedWaterActive);
        if (!hasWater)
        {
            // 沙运动后的容量修正对空水场恒为恒等映射。清理缓存可保持诊断输出
            // 与完整路径一致，同时避免干沙场景每个子步重复遍历水网格。
            state.pureWater.velocity.SetZero();
            state.mixedWater.velocity.SetZero();
            scratch.mixedHeightDelta.SetZero();
            scratch.waterTransferImpulse.SetZero();
            return report;
        }
        report.totalVolumeBefore = ComputeTotalWaterVolume(state, boundary);
        report.totalVolumeAfterDiffusion = report.totalVolumeBefore;
        report.totalVolumeAfterAdvection = report.totalVolumeBefore;
        report.mixedVolumeBeforeDiffusion =
            ComputeVolume(state.mixedWater.height, boundary);
        report.mixedVolumeAfterDiffusion =
            report.mixedVolumeBeforeDiffusion;

        CorrectLayerHeightsAndBuildImpulse(
            state, config, dt, boundary, scratch, report);
        report.totalVolumeAfterLayerCorrection =
            ComputeTotalWaterVolume(state, boundary);

        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);
        boundary.BuildWaterTerrainOpenMask(
            state,
            config.time.heightEpsilon,
            scratch.waterTerrainFaceOpen);
        report.maxLayerMomentumError =
            ApplyLayerTransferImpulse(state, config, boundary, scratch);

        boundary.BuildFaceMask(scratch.pureWaterActive, scratch.faceValid);
        boundary.IntersectFaceMasks(
            scratch.faceValid,
            scratch.waterTerrainFaceOpen,
            scratch.faceValid);
        FinalizeVelocity(
            state.pureWater, config.time.maxVelocity, boundary, scratch.faceValid);
        boundary.BuildFaceMask(scratch.mixedWaterActive, scratch.faceValid);
        boundary.IntersectFaceMasks(
            scratch.faceValid,
            scratch.waterTerrainFaceOpen,
            scratch.faceValid);
        FinalizeVelocity(
            state.mixedWater, config.time.maxVelocity, boundary, scratch.faceValid);
        boundary.UpdateMasks(state, config.time.heightEpsilon, scratch);

        report.totalVolumeAfterStep = ComputeTotalWaterVolume(state, boundary);
        report.minHeight = ComputeMinWaterHeight(state, boundary);
        return report;
    }

    // ----- mixed-water diffusion -------------------------------------------------

    void WaterSolver::DiffuseMixedWater(
        LayerState &mixedWater,
        SimulationConfig const &config,
        double dt,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        if (config.physics.diffusionRate == 0.0)
        {
            // 零扩散的解析结果就是输入本身。调用点已经重施高度/速度边界，
            // 因此无需为恒等算子复制三张场并启动两个并行 stencil。
            return;
        }
        // 两个快照必须在任一 destination 被写入前完成。否则并行循环会读取
        // 已扩散的邻居，结果依赖执行顺序并破坏 Neumann 质量守恒。
        scratch.scalar0 = mixedWater.height;
        scratch.velocity0 = mixedWater.velocity;
        DiffuseMixedHeight(
            scratch.scalar0,
            config.physics.diffusionRate,
            dt,
            boundary,
            scratch.scalar1);
        DiffuseMixedVelocity(
            scratch.scalar0,
            scratch.velocity0,
            config.physics.diffusionRate,
            dt,
            boundary,
            mixedWater.velocity);

        mixedWater.height = scratch.scalar1;
        boundary.ApplyDiffusionNeumann(mixedWater.height);
        boundary.ApplyVelocityBoundary(mixedWater.velocity);
    }

    void WaterSolver::DiffuseMixedHeight(
        GridData<double> const &source,
        double diffusionRate,
        double dt,
        BoundaryConditions const &boundary,
        GridData<double> &destination) const
    {
        double const coefficient =
            diffusionRate * dt *
            source.GetGrid().GetInvSpacing() *
            source.GetGrid().GetInvSpacing();
        if (coefficient > 0.25 +
                              64.0 * std::numeric_limits<double>::epsilon())
        {
            throw std::runtime_error(
                "mixed-water diffusion violates the 2D explicit stability limit");
        }

        destination = source;
        ParallelForEach(source.GetGrid(), [&](Vector2i const &cell)
                        {
            if (!boundary.IsInsideCell(cell))
            {
                return;
            }

            double laplaceStencil = -4.0 * source[cell];
            for (int ordinal = 0; ordinal < Grid::GetNumNeighbors(); ++ordinal)
            {
                Vector2i const neighbor = Grid::NeighborOf(cell, ordinal);
                // source 的 boundary band 已由 Neumann 条件从最近内部单元复制。
                laplaceStencil += source[neighbor];
            }
            double const updated = source[cell] + coefficient * laplaceStencil;
            if (updated < -64.0 * std::numeric_limits<double>::epsilon() *
                              std::max(1.0, std::abs(source[cell])))
            {
                throw std::runtime_error(fmt::format(
                    "mixed-water diffusion produced negative height {} at ({}, {})",
                    updated, cell.x(), cell.y()));
            }
            destination[cell] = std::max(updated, 0.0);
        });
    }

    void WaterSolver::DiffuseMixedVelocity(
        GridData<double> const &oldHeight,
        SGridData<double> const &oldVelocity,
        double diffusionRate,
        double dt,
        BoundaryConditions const &boundary,
        SGridData<double> &destination) const
    {
        destination = oldVelocity;
        double const dx = oldHeight.GetGrid().GetSpacing();
        double const coefficient =
            diffusionRate * dt /
            (dx * dx);

        ParallelForEach(oldVelocity.GetGrids(), [&](int axis, Vector2i const &face)
                        {
            if (!boundary.IsInsideFace(axis, face) ||
                boundary.IsNormalBoundaryFace(axis, face))
            {
                destination[axis][face] = 0.0;
                return;
            }

            // 论文 4.1.1：在当前 MAC 面周围，对每个方向累加
            // v(x±dx/2 e_d) * [h(x±dx e_d)-h(x)]。高度和速度位于不同
            // 拓扑位置，因此都通过双线性插值取到公式要求的位置。
            Vector2d const position =
                oldVelocity[axis].GetGrid().PositionOf(face);
            double const centerHeight =
                BiLerp::Interpolate(oldHeight, position);
            double increment = 0.0;
            for (int direction = 0; direction < 2; ++direction)
            {
                Vector2d const offset =
                    dx * Vector2d::Unit(direction);
                double const heightPlus =
                    BiLerp::Interpolate(oldHeight, position + offset);
                double const heightMinus =
                    BiLerp::Interpolate(oldHeight, position - offset);
                double const velocityPlus = BiLerp::Interpolate(
                    oldVelocity[axis], position + 0.5 * offset);
                double const velocityMinus = BiLerp::Interpolate(
                    oldVelocity[axis], position - 0.5 * offset);
                increment +=
                    velocityPlus * (heightPlus - centerHeight) -
                    velocityMinus * (centerHeight - heightMinus);
            }
            destination[axis][face] =
                oldVelocity[axis][face] + coefficient * increment;
        });
    }

    // ----- conservative height advection ----------------------------------------

    AdvectionStats WaterSolver::AdvectHeight(
        LayerState &water,
        double dt,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        AdvectionStats const stats = AdvectionFV::SolveHeight(
            water.height,
            water.velocity,
            dt,
            boundary,
            scratch.scalar0);
        water.height = scratch.scalar0;
        boundary.ApplyHeightBoundary(water.height);
        return stats;
    }

    // ----- pure/mixed layer correction ------------------------------------------

    void WaterSolver::CorrectLayerHeightsAndBuildImpulse(
        SimulationState &state,
        SimulationConfig const &config,
        double,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch,
        WaterStepReport &report) const
    {
        scratch.waterTransferImpulse.SetZero();
        double maxTransfer = 0.0;

        ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
                        {
            if (!boundary.IsInsideCell(cell))
            {
                scratch.totalWater[cell] = 0.0;
                scratch.mixedHeightDelta[cell] = 0.0;
                return;
            }

            double const predictedPure =
                std::max(state.pureWater.height[cell], 0.0);
            double const predictedMixed =
                std::max(state.mixedWater.height[cell], 0.0);
            double const total = predictedPure + predictedMixed;
            double const sandCapacity =
                std::max(state.sand.height[cell], 0.0);
            double const correctedMixed = std::min(total, sandCapacity);
            double const correctedPure = total - correctedMixed;

            scratch.totalWater[cell] = total;
            scratch.mixedHeightDelta[cell] =
                correctedMixed - predictedMixed;
            state.mixedWater.height[cell] = correctedMixed;
            state.pureWater.height[cell] = correctedPure;
        });

        // maxAbs 只用于诊断，串行归约避免在普通 double 上使用原子操作。
        ForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
                {
            if (boundary.IsInsideCell(cell))
            {
                maxTransfer = std::max(
                    maxTransfer,
                    std::abs(scratch.mixedHeightDelta[cell]));
            }
        });
        report.maxAbsLayerTransfer =
            std::max(report.maxAbsLayerTransfer, maxTransfer);

        boundary.ApplyHeightBoundary(state.pureWater.height);
        boundary.ApplyHeightBoundary(state.mixedWater.height);
        boundary.ApplyHeightBoundary(scratch.totalWater);
        boundary.ApplyHeightBoundary(scratch.mixedHeightDelta);

        // 保存论文意义下的有符号 donor 冲量密度 rho*deltaH*v_donor。
        // 实际有限更新在速度平流/压力之后用相同 donor 规则重组，以避免薄层除法。
        ParallelForEach(state.pureWater.velocity.GetGrids(),
                        [&](int axis, Vector2i const &face)
                        {
                            if (!boundary.IsInsideFace(axis, face) ||
                                boundary.IsNormalBoundaryFace(axis, face))
                            {
                                scratch.waterTransferImpulse[axis][face] = 0.0;
                                return;
                            }
                            double const delta = AverageInteriorCellsToFace(
                                scratch.mixedHeightDelta, axis, face, boundary);
                            double const donorVelocity = delta >= 0.0
                                                             ? state.pureWater.velocity[axis][face]
                                                             : state.mixedWater.velocity[axis][face];
                            scratch.waterTransferImpulse[axis][face] =
                                config.physics.waterDensity * delta * donorVelocity;
                        });
    }

    double WaterSolver::ApplyLayerTransferImpulse(
        SimulationState &state,
        SimulationConfig const &config,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        double maxError = 0.0;
        ForEach(state.pureWater.velocity.GetGrids(),
                [&](int axis, Vector2i const &face)
                {
                    if (!boundary.IsInsideFace(axis, face) ||
                        boundary.IsNormalBoundaryFace(axis, face))
                    {
                        state.pureWater.velocity[axis][face] = 0.0;
                        state.mixedWater.velocity[axis][face] = 0.0;
                        return;
                    }

                    double const delta = AverageInteriorCellsToFace(
                        scratch.mixedHeightDelta, axis, face, boundary);
                    double const correctedPure = AverageInteriorCellsToFace(
                        state.pureWater.height, axis, face, boundary);
                    double const correctedMixed = AverageInteriorCellsToFace(
                        state.mixedWater.height, axis, face, boundary);
                    // 由 corrected = predicted ± delta 精确恢复 predictor 面高度。
                    double const predictedPure = std::max(correctedPure + delta, 0.0);
                    double const predictedMixed = std::max(correctedMixed - delta, 0.0);

                    double const pureVelocity =
                        state.pureWater.velocity[axis][face];
                    double const mixedVelocity =
                        state.mixedWater.velocity[axis][face];
                    double const momentumBefore =
                        predictedPure * pureVelocity +
                        predictedMixed * mixedVelocity;

                    double newPureVelocity = pureVelocity;
                    double newMixedVelocity = mixedVelocity;
                    if (delta > 0.0)
                    {
                        // pure -> mixed：纯水移出的质量携带 pureVelocity，接收层按
                        // 旧混合水动量与 donor 动量之和重组。
                        newMixedVelocity = correctedMixed > 0.0
                                               ? (predictedMixed * mixedVelocity +
                       delta * pureVelocity) / correctedMixed
                                               : 0.0;
                    }
                    else if (delta < 0.0)
                    {
                        // mixed -> pure：符号取反后的 transferredHeight 是正质量。
                        double const transferredHeight = -delta;
                        newPureVelocity = correctedPure > 0.0
                                              ? (predictedPure * pureVelocity +
                       transferredHeight * mixedVelocity) / correctedPure
                                              : 0.0;
                    }

                    if (correctedPure <= config.time.heightEpsilon)
                    {
                        newPureVelocity = 0.0;
                    }
                    if (correctedMixed <= config.time.heightEpsilon)
                    {
                        newMixedVelocity = 0.0;
                    }
                    state.pureWater.velocity[axis][face] = newPureVelocity;
                    state.mixedWater.velocity[axis][face] = newMixedVelocity;

                    double const momentumAfter =
                        correctedPure * newPureVelocity +
                        correctedMixed * newMixedVelocity;
                    maxError = std::max(
                        maxError, std::abs(momentumAfter - momentumBefore));
                });
        return config.physics.waterDensity * maxError;
    }

    // ----- semi-Lagrangian velocity advection -----------------------------------

    void WaterSolver::AdvectVelocity(
        LayerState &water,
        SGridData<std::uint8_t> const &sourceValidFaces,
        double dt,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        // 复制源速度是必要的：半拉格朗日采样必须全部读取 n 时刻，而 destination
        // 会在另一个场中并行写入。两圈外推给中点回溯提供可采样的干区速度。
        scratch.velocity0 = water.velocity;
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
            water.velocity);
        boundary.ApplyVelocityBoundary(water.velocity);
    }

    // ----- hydrostatic pressure --------------------------------------------------

    void WaterSolver::BuildHydraulicPotential(
        SimulationState const &state,
        BoundaryConditions const &boundary,
        SimulationScratch &scratch) const
    {
        ParallelForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
                        {
            // 保留 mixedWater 项使该势函数在 M2 接入两层水时无需改变定义。
            scratch.hydraulicPotential[cell] =
                state.terrain[cell] +
                state.pureWater.height[cell] +
                state.mixedWater.height[cell]; });
        boundary.ApplyHeightBoundary(scratch.hydraulicPotential);
    }

    void WaterSolver::ApplyHydrostaticPressure(
        LayerState &water,
        GridData<double> const &level,
        double gravity,
        double dt,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &validFaces) const
    {
        ParallelForEach(water.velocity.GetGrids(), [&](int axis, Vector2i const &face)
                        {
            if (!validFaces[axis][face] ||
                boundary.IsNormalBoundaryFace(axis, face) ||
                !boundary.IsInsideFace(axis, face))
            {
                water.velocity[axis][face] = 0.0;
                return;
            }

            water.velocity[axis][face] -=
                gravity * dt * MacOperators::GradientAtFace(level, axis, face); });
    }

    void WaterSolver::FinalizeVelocity(
        LayerState &water,
        double maxVelocity,
        BoundaryConditions const &boundary,
        SGridData<std::uint8_t> const &validFaces) const
    {
        ParallelForEach(water.velocity.GetGrids(), [&](int axis, Vector2i const &face)
                        {
            if (!validFaces[axis][face] || !boundary.IsInsideFace(axis, face))
            {
                water.velocity[axis][face] = 0.0;
                return;
            }
            water.velocity[axis][face] =
                std::clamp(water.velocity[axis][face], -maxVelocity, maxVelocity); });
        boundary.ApplyVelocityBoundary(water.velocity);
    }

    double WaterSolver::ComputeVolume(
        GridData<double> const &height,
        BoundaryConditions const &boundary)
    {
        double const area =
            height.GetGrid().GetSpacing() * height.GetGrid().GetSpacing();
        double volume = 0.0;
        ForEach(height.GetGrid(), [&](Vector2i const &cell)
                {
            if (boundary.IsInsideCell(cell))
            {
                volume += height[cell] * area;
            } });
        return volume;
    }

    double WaterSolver::ComputeTotalWaterVolume(
        SimulationState const &state,
        BoundaryConditions const &boundary)
    {
        return ComputeVolume(state.pureWater.height, boundary) +
               ComputeVolume(state.mixedWater.height, boundary);
    }

    double WaterSolver::ComputeMinWaterHeight(
        SimulationState const &state,
        BoundaryConditions const &boundary)
    {
        double minimum = std::numeric_limits<double>::infinity();
        ForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
                {
            if (boundary.IsInsideCell(cell))
            {
                minimum = std::min({
                    minimum,
                    state.pureWater.height[cell],
                    state.mixedWater.height[cell]});
            }
        });
        return std::isfinite(minimum) ? minimum : 0.0;
    }
}
