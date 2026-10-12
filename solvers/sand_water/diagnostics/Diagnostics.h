#pragma once

#include "numerics/BoundaryConditions.h"
#include "simulation/SimulationConfig.h"

namespace Pivot
{
    /// 一帧模拟的全局守恒量和稳定性摘要。
    ///
    /// 体积由内部单元的“高度 × 单元面积”积分得到，边界带不参与统计。
    struct ConservationReport
    {
        /// 沙、纯水、混合水以及两种水之和的总体积。
        double sandVolume{0.0};
        double pureWaterVolume{0.0};
        double mixedWaterVolume{0.0};
        double totalWaterVolume{0.0};
        /// 三个材料层的二维水平动量之和。
        Vector2d horizontalMomentum{Vector2d::Zero()};
        /// 所有材料高度的最小值，用于发现负高度。
        double minHeight{0.0};
        /// 活跃沙单元中 det(F) 的最小值；无沙时约定为 1。
        double minDetF{1.0};
        /// 最大屈服面残差；M3 尚未接入塑性模型，因此保持为 0。
        double maxYieldResidual{0.0};
        /// 三个材料层所有 MAC 面上的最大速度绝对值。
        double maxVelocity{0.0};
        /// 最近一步实际使用的时间步；初始帧为零。
        double timeStep{0.0};
        /// 以最近实际时间步计算的 maxVelocity * dt / dx。
        double cfl{0.0};
        /// 最近水步中混合水扩散、两层平流和容量修正造成的体积变化。
        ///
        /// 它们是子算子级诊断而非新的守恒量；初始帧均为零。
        double mixedDiffusionVolumeChange{0.0};
        double waterAdvectionVolumeChange{0.0};
        double layerCorrectionVolumeChange{0.0};
        /// 最近水步的最大层转换高度和面心动量重组舍入误差。
        double maxLayerTransfer{0.0};
        double maxLayerMomentumError{0.0};
        /// 最近外层时间步中全部沙子步的平流质量差和最大加速度。
        double sandAdvectionVolumeChange{0.0};
        double maxSandGravityAcceleration{0.0};
        /// M5 阻力交换的相对速度、冲量及逐面动量守恒诊断。
        double maxExchangeRelativeSpeedBefore{0.0};
        double maxExchangeRelativeSpeedAfter{0.0};
        double maxExchangeImpulse{0.0};
        double maxExchangeMomentumError{0.0};
        std::uint64_t exchangedFaceCount{0};
        /// 相对速度经一步交换后反号的面数；解析隐式模式应始终为零。
        std::uint64_t exchangeOvershootFaceCount{0};
        /// 最近沙子步中弹塑性应力散度产生的最大加速度。
        double maxSandInternalAcceleration{0.0};
        /// 上述最大内力所在面的真实沙高，以及法向/切向模板各自峰值。
        double sandHeightAtMaxInternalAcceleration{0.0};
        double maxSandNormalStressAcceleration{0.0};
        double maxSandTangentialStressAcceleration{0.0};
        /// 可选平流后返回映射的本构一致性统计；开关关闭时均为零。
        double maxPostAdvectionYieldResidualBefore{0.0};
        double maxPostAdvectionYieldResidualAfter{0.0};
        double maxPostAdvectionProjectionDelta{0.0};
        std::uint64_t postAdvectionProjectionCount{0};
        /// 数值保护执行之前的沙速度峰值和实际触发的限幅事件数。
        double maxSandVelocityBeforeClamping{0.0};
        std::uint64_t sandVelocityClampCount{0};
        double maxSandFrictionDeceleration{0.0};
        /// Coulomb 摩擦在最近外层步中完全停止的面数量。
        std::uint64_t frictionStoppedFaces{0};
        /// 沙速度按实际 sandDt=waterDt/sandSubsteps 计算的 CFL。
        double sandCfl{0.0};
    };

    /// 无状态的运行时诊断服务。
    ///
    /// Measure() 提供可绘图的连续指标；Validate() 负责 fail-fast 不变量检查。
    /// 两者分离，使发布版本可按需要降低昂贵检查的调用频率。
    class Diagnostics
    {
    public:
        /// 对内部物理域积分守恒量，并统计全局极值。
        ConservationReport Measure(
            SimulationState const &state,
            SimulationConfig const &config,
            StaggeredGrid const &grid) const;

        /// 验证状态是否有限、材料高度是否合法、F 是否可逆以及封闭边界是否满足。
        /// stage 会写入异常消息，便于定位是哪一个时间步阶段首先破坏了不变量。
        void Validate(
            SimulationState const &state,
            SimulationConfig const &config,
            StaggeredGrid const &grid,
            BoundaryConditions const &boundary,
            std::string_view stage) const;
    };
}
