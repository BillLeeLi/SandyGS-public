#pragma once

#include "Advection.h"
#include "Extrapolation.h"
#include "numerics/MacOperators.h"
#include "physics/SandConstitutive.h"
#include "simulation/SimulationConfig.h"

namespace Pivot
{
    /// 一个外层水步内全部沙子步的局部数值诊断。
    ///
    /// Simulation 会把每个 sand substep 的统计累积到同一报告中，使输出曲线
    /// 能区分沙高平流误差、重力加速度和摩擦耗散，而不只观察最终沙体积。
    struct SandStepReport
    {
        double volumeBefore{0.0};
        double volumeAfterStep{0.0};
        double cumulativeAdvectionVolumeChange{0.0};
        double minHeight{std::numeric_limits<double>::infinity()};
        double maxGravityAcceleration{0.0};
        /// M5 沙—混合水阻力交换前后的最大面心相对速度分量。
        double maxExchangeRelativeSpeedBefore{0.0};
        double maxExchangeRelativeSpeedAfter{0.0};
        /// 单个面从沙传给混合水的最大冲量绝对值（单位水平面积）。
        double maxExchangeImpulse{0.0};
        /// 交换前后两相离散柱动量的最大舍入误差。
        double maxExchangeMomentumError{0.0};
        /// 实际同时含有沙和混合水的交换面数；多个沙子步会累加。
        std::uint64_t exchangedFaceCount{0};
        /// 论文显式顺序下相对速度跨过零点的面数，用于暴露过大的交换步长。
        std::uint64_t exchangeOvershootFaceCount{0};
        /// h*sigma 的离散散度产生的最大面加速度。
        double maxInternalAcceleration{0.0};
        /// 最大内力加速度所在面的沙高，用于识别薄层除法放大。
        double heightAtMaxInternalAcceleration{
            std::numeric_limits<double>::infinity()};
        /// 将 div(h*sigma) 拆成法向与切向模板后的各自最大贡献。
        double maxNormalStressAcceleration{0.0};
        double maxTangentialStressAcceleration{0.0};
        /// 可选的“平流后返回映射”在投影前后的屈服残差峰值。
        double maxPostAdvectionYieldResidualBefore{0.0};
        double maxPostAdvectionYieldResidualAfter{0.0};
        /// 新返回映射对单元 F 的最大 Frobenius 改变量，以及累计修改次数。
        double maxPostAdvectionProjectionDelta{0.0};
        std::uint64_t postAdvectionProjectionCount{0};
        /// 速度保护触发前的最大绝对分量以及本外层步的限幅事件数。
        double maxVelocityBeforeClamping{0.0};
        std::uint64_t velocityClampCount{0};
        double maxFrictionDeceleration{0.0};
        std::uint64_t frictionStoppedFaces{0};

        double VolumeChange() const
        {
            return volumeAfterStep - volumeBefore;
        }
    };

    /// 集中管理浅层沙平流、柱重力、弹塑性内力和地面摩擦的无状态求解器。
    ///
    /// M6.6 修正后的固定顺序为：
    ///   塑性投影 -> h/theta/P 同通量平流 -> 沙速度平流/重力
    ///   -> 沙/混合水动量交换 -> div(h*sigma) 内力。
    /// 水层容量修正由 Simulation 插入后，ApplyFrictionAndFinalize() 先得到
    /// 含 Coulomb 摩擦的最终速度，再用该速度更新并返回映射 F。
    class SandSolver
    {
    public:
        /// 执行一个沙子步中层修正之前的动力学阶段。
        ///
        /// report 由 Simulation 在外层水步开始时初始化，本方法把当前子步的
        /// 平流质量差和最大重力加速度累积进去。
        void IntegrateDynamics(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions &boundary,
            SimulationScratch &scratch,
            SandStepReport &report,
            bool projectInputState = true) const;

        /// 在水层重划分后施加摩擦，并以最终速度提交本步 F 的运动学更新。
        void ApplyFrictionAndFinalize(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions &boundary,
            SimulationScratch &scratch,
            SandStepReport &report) const;

    private:
        struct ProjectionStats
        {
            double maximumYieldResidualBefore{0.0};
            double maximumYieldResidualAfter{0.0};
            double maximumDeformationDelta{0.0};
            std::uint64_t changedCellCount{0};
        };

        /// 一次内力散度计算的分量级统计。
        struct InternalAccelerationStats
        {
            double maximum{0.0};
            double heightAtMaximum{
                std::numeric_limits<double>::infinity()};
            double maximumNormal{0.0};
            double maximumTangential{0.0};
        };

        /// 用沙速度守恒平流沙高，并返回该子步的体积变化。
        AdvectionStats AdvectHeight(
            SandState &sand,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 从高度更新前的有效面向新沙面外推速度，再作中点回溯平流。
        void AdvectVelocity(
            SandState &sand,
            SGridData<std::uint8_t> const &sourceValidFaces,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 投影 active F，并以连续 theta 与 P 向沙边界外做一圈 3x3 外推。
        ProjectionStats ProjectDeformationGradient(
            SimulationState &state,
            SimulationConfig const &config,
            SimulationScratch &scratch,
            bool capturePostAdvectionDiagnostics = false) const;

        /// 重建本构使用的饱和度；可选低通只改变本构输入，不改写物理状态。
        void BuildConstitutiveSaturation(
            SimulationState const &state,
            SimulationConfig const &config,
            SimulationScratch &scratch) const;

        /// 用旧 h 构造 h*theta/h*P，并与沙高使用同一有限体积通量更新。
        void AdvectConstitutiveState(
            SandState &sand,
            GridData<double> const &oldHeight,
            SGridData<double> const &oldVelocity,
            double dt,
            BoundaryConditions const &boundary,
            SimulationConfig const &config,
            SimulationScratch &scratch) const;

        /// 计算饱和度、Cauchy 应力以及 Q=h*sigma。
        void ComputeStress(
            SimulationState const &state,
            SimulationConfig const &config,
            SimulationScratch &scratch) const;

        /// 把 div(h*sigma)/(rho_s*h_face) 离散到 MAC 面。
        InternalAccelerationStats ComputeInternalAcceleration(
            GridData<double> const &sandHeight,
            GridData<Matrix2d> const &weightedStress,
            double sandDensity,
            double heightEpsilon,
            double heightFloor,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &validFaces,
            SGridData<double> &acceleration,
            SGridData<double> &normalAcceleration,
            SGridData<double> &tangentialAcceleration,
            SGridData<double> &faceHeight,
            SGridData<double> &effectiveHeight) const;

        /// 计算 div(h*sigma)/(rho_s*h)，并把内力增量加入候选沙速度。
        void ApplyInternalForce(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch,
            SandStepReport &report) const;

        /// 用经过全部力和摩擦后的速度按 DF/Dt=grad(v)F 更新 active F。
        void EvolveDeformationGradient(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            SimulationScratch const &scratch) const;

        /// 从 theta/P 重建二维 F=R(theta)P。
        static Matrix2d ReconstructDeformationGradient(
            double rotationAngle,
            Matrix2d const &stretch);

        /// 构造 Ψs = H + hs + (ρw/ρs) hpure；混合水不直接进入沙重力势。
        void BuildSandPotential(
            SimulationState const &state,
            SimulationConfig const &config,
            BoundaryConditions const &boundary,
            SimulationScratch &scratch) const;

        /// 在有效沙面施加 -g*dt*grad(Ψs)，返回本子步最大加速度绝对值。
        double ApplyGravityAndWaterLoad(
            SandState &sand,
            GridData<double> const &potential,
            double gravity,
            double dt,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &validFaces) const;

        /// 在共同占据的 MAC 面交换沙与混合水动量。
        ///
        /// 两种离散都把同一个有符号冲量从沙柱扣除并加入水柱，因此逐面严格
        /// 守恒；PaperSequential 对应论文式 4.2.3，AnalyticImplicit 则解析
        /// 求解一步线性阻力以避免相对速度过冲。
        void ExchangeMomentumWithMixedWater(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &sandValidFaces,
            SimulationScratch &scratch,
            SandStepReport &report) const;

        /// 按论文 4.2.4 对完整二维速度统一缩放，保持摩擦前后的速度方向。
        void ApplyFriction(
            SimulationState &state,
            SimulationConfig const &config,
            double dt,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &validFaces,
            SimulationScratch &scratch,
            SandStepReport &report) const;

        /// 清除干面速度、逐分量限幅并重施封闭边界。
        void FinalizeVelocity(
            SandState &sand,
            double maxVelocity,
            BoundaryConditions const &boundary,
            SGridData<std::uint8_t> const &validFaces,
            SandStepReport &report) const;

        static double ComputeVolume(
            GridData<double> const &height,
            BoundaryConditions const &boundary);

        static double ComputeMinHeight(
            GridData<double> const &height,
            BoundaryConditions const &boundary);
    };
}
