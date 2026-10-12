#pragma once

#include "Advection.h"
#include "Extrapolation.h"
#include "numerics/MacOperators.h"
#include "simulation/SimulationConfig.h"

namespace Pivot
{
    /// 一个水大步的局部守恒诊断。
    ///
    /// 各阶段体积单独记录，使质量误差能定位到扩散、平流或层间重新划分，
    /// 而不只在完整时间步末看到一个总误差。
    struct WaterStepReport
    {
        /// 时间步开始及各子阶段结束时的总水体积。
        double totalVolumeBefore{0.0};
        double totalVolumeAfterDiffusion{0.0};
        double totalVolumeAfterAdvection{0.0};
        double totalVolumeAfterLayerCorrection{0.0};
        double totalVolumeAfterStep{0.0};
        /// 扩散前后的混合水体积；封闭 Neumann 域中二者应相等。
        double mixedVolumeBeforeDiffusion{0.0};
        double mixedVolumeAfterDiffusion{0.0};
        /// 单个单元在本步发生的最大纯水/混合水转换高度。
        double maxAbsLayerTransfer{0.0};
        /// 层转换前后面心“高度 × 速度”总量的最大舍入误差。
        double maxLayerMomentumError{0.0};
        double minHeight{0.0};

        double DiffusionVolumeChange() const
        {
            return mixedVolumeAfterDiffusion - mixedVolumeBeforeDiffusion;
        }

        double AdvectionVolumeChange() const
        {
            return totalVolumeAfterAdvection - totalVolumeAfterDiffusion;
        }

        double LayerCorrectionVolumeChange() const
        {
            return totalVolumeAfterLayerCorrection - totalVolumeAfterAdvection;
        }
    };

    /// 集中管理水相数值更新的无状态求解器。
    ///
    /// M2 的固定顺序为：
    ///   混合水扩散 -> 两层高度守恒平流 -> 层容量修正 -> 两层速度平流
    ///   -> 共同静水压力 -> donor 动量重组。
    /// 所有子算子仍集中在本类型中，避免为一次水更新拆出大量小文件。
    class WaterSolver
    {
    public:
        /// 将水状态推进 dt，并返回该步的水体积诊断。
        ///
        /// state/scratch 中的所有场必须绑定到 boundary 所引用的同一网格。
        WaterStepReport Step(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions &boundary,
            SimulationScratch &scratch) const;

        /// 沙高改变后立即恢复 mixedWater.height <= sand.height。
        ///
        /// M3 的每个沙子步改变沙高后由 Simulation 调用；转换动量在本函数
        /// 内立即应用，使 mixedWater.height <= sand.height 不延迟到下一水步。
        WaterStepReport CorrectLayersAfterSandMotion(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions &boundary,
            SimulationScratch &scratch) const;

    private:
        /// 同步扩散混合水高度和论文 4.1.1 的面心速度增量。
        void DiffuseMixedWater(
            LayerState &mixedWater,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 使用旧高度的五点 Laplacian 写入 destination，关闭边界为 Neumann。
        void DiffuseMixedHeight(
            GridData<double> const &source,
            double diffusionRate,
            double dt,
            BoundaryConditions const &boundary,
            GridData<double> &destination) const;

        /// 按论文面心离散式计算扩散所诱导的混合水速度变化。
        void DiffuseMixedVelocity(
            GridData<double> const &oldHeight,
            SGridData<double> const &oldVelocity,
            double diffusionRate,
            double dt,
            BoundaryConditions const &boundary,
            SGridData<double> &destination) const;

        /// 用该层自身速度守恒平流其高度。
        AdvectionStats AdvectHeight(
            LayerState &water,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 按沙柱容量重新划分两层水，并保存每个单元的 mixed 高度增量。
        void CorrectLayerHeightsAndBuildImpulse(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch,
            WaterStepReport &report) const;

        /// 用 donor 相速度重组接收相动量；返回面心最大动量守恒误差。
        double ApplyLayerTransferImpulse(
            SimulationState &state,
            SimulationConfig const &config,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 将有效水面速度外推后，用自身流场作中点回溯平流。
        void AdvectVelocity(
            LayerState &water,
            SGridData<std::uint8_t> const &sourceValidFaces,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 构造水压力势 level = terrain + pureWater + mixedWater。
        void BuildHydraulicPotential(
            SimulationState const &state,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 对有效水面施加 -g*dt*grad(level)，平坦自由液面应保持静止。
        void ApplyHydrostaticPressure(
            LayerState &water,
            GridData<double> const &level,
            double gravity,
            double dt,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &validFaces) const;

        /// 清除完全干燥面速度，并对湿面速度逐分量限幅。
        void FinalizeVelocity(
            LayerState &water,
            double maxVelocity,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &validFaces) const;

        static double ComputeVolume(
            GridData<double> const &height,
            BoundaryConditions const &boundary);

        static double ComputeTotalWaterVolume(
            SimulationState const &state,
            BoundaryConditions const &boundary);

        static double ComputeMinWaterHeight(
            SimulationState const &state,
            BoundaryConditions const &boundary);
    };
}
