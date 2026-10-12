#pragma once

#include "SGridData.h"
#include "StaggeredGrid.h"

namespace Pivot
{
    /// 一个高度场材料层的最小动态状态。
    ///
    /// height 位于单元中心，velocity 的两个分量分别位于 MAC 网格的 x/y 面。
    /// SandState、纯水层和沙内混合水层均复用这一布局。
    struct LayerState
    {
        /// 材料柱高 h(i,j)。
        GridData<double> height;
        /// 水平方向面速度；velocity[axis] 对应法向为 axis 的网格面。
        SGridData<double> velocity;

        explicit LayerState(StaggeredGrid const &grid);
        /// 将高度和速度清零。
        void SetZero();
    };

    /// 沙层状态：通用层变量加上弹塑性本构变量。
    struct SandState : LayerState
    {
        /// 单元中心的二维变形梯度 F。
        GridData<Matrix2d> deformationGradient;
        /// 极分解 F = R(theta) P 中旋转矩阵 R 的角度。
        GridData<double> rotationAngle;
        /// 极分解中的对称拉伸矩阵 P。
        GridData<Matrix2d> stretch;

        explicit SandState(StaggeredGrid const &grid);
        /// 清空动力学量，并把 F/P 重置为单位阵。
        void SetZero();
        /// 将无沙单元的本构变量恢复为无形变状态，防止旧值污染后续激活。
        void ResetConstitutiveState(GridData<std::uint8_t> const &active);
    };

    /// 某个时间点需要永久保留的全部模拟状态。
    ///
    /// 该类型拥有绑定到同一 StaggeredGrid 的数据场，因 GridData 内部保存网格
    /// 引用，所以禁止复制和移动。算法的中间结果统一放在 SimulationScratch。
    class SimulationState
    {
    public:
        explicit SimulationState(StaggeredGrid const &grid);

        SimulationState(SimulationState const &) = delete;
        SimulationState &operator=(SimulationState const &) = delete;
        SimulationState(SimulationState &&) = delete;
        SimulationState &operator=(SimulationState &&) = delete;

        /// 恢复 t=0 的空场状态；地形和所有材料数据都会被重置。
        void SetZero();

        /// 固定地形高度 H。
        GridData<double> terrain;
        /// 可运动沙层。
        SandState sand;
        /// 位于沙层上方、不受孔隙容量限制的自由水层。
        LayerState pureWater;
        /// 位于沙层内部的水层，必须始终满足 mixedWater.height <= sand.height。
        LayerState mixedWater;

        /// 当前物理时间、最近实际步长和已完成的外层时间步编号。
        double time{0.0};
        double lastDt{0.0};
        std::uint64_t stepIndex{0};
    };

    /// 可跨算子复用的工作缓存，不属于可保存的物理状态。
    ///
    /// 提前分配这些场可避免每个时间步频繁申请内存。带明确名称的字段保留给
    /// 固定用途；scalar0/matrix0/velocity0 等字段供单个算法短期复用，调用者
    /// 不得假设它们会跨阶段保留值。
    class SimulationScratch
    {
    public:
        explicit SimulationScratch(StaggeredGrid const &grid);

        SimulationScratch(SimulationScratch const &) = delete;
        SimulationScratch &operator=(SimulationScratch const &) = delete;
        SimulationScratch(SimulationScratch &&) = delete;
        SimulationScratch &operator=(SimulationScratch &&) = delete;

        /// 清零全部临时场和有效性掩码。
        void SetZero();

        // 通用标量缓存，以及水层拆分/势能计算所需的专用缓存。
        GridData<double> scalar0;
        GridData<double> scalar1;
        /// pureWater + mixedWater，用于守恒的水层重新分配。
        GridData<double> totalWater;
        /// 本步从纯水层转入/移出混合水层的高度增量。
        GridData<double> mixedHeightDelta;
        /// 沙平流或耦合更新前的沙高度快照。
        GridData<double> oldSandHeight;
        /// 沙层局部含水状态；后续本构和黏聚力模型共享该场。
        GridData<double> saturation;
        /// 未经低通的本构饱和度，只用于与 saturation 做诊断对照。
        GridData<double> rawConstitutiveSaturation;
        /// 驱动水流和沙流的单元中心势函数。
        GridData<double> hydraulicPotential;
        GridData<double> sandPotential;

        // 本构分解、应力计算和加权平均使用的矩阵缓存。
        GridData<Matrix2d> matrix0;
        GridData<Matrix2d> matrix1;
        GridData<Matrix2d> stress;
        GridData<Matrix2d> weightedStress;
        /// 最近一次沙内力计算实际使用的屈服残差；区别于帧末按最终 F 重算值。
        GridData<double> internalYieldResidual;
        /// 可选的平流后返回映射在每个单元投影前/后的屈服残差。
        GridData<double> postAdvectionYieldResidualBefore;
        GridData<double> postAdvectionYieldResidualAfter;
        /// 平流后返回映射造成的 ||F_after-F_before||_F；零表示没有修改。
        GridData<double> postAdvectionProjectionDelta;

        // 面中心速度/冲量/加速度缓存。
        SGridData<double> velocity0;
        /// 纯水与混合水重新分配时携带的动量。
        SGridData<double> waterTransferImpulse;
        /// 沙层和水层之间的双向作用冲量。
        SGridData<double> sandWaterImpulse;
        /// 由压力或沙体内应力产生的面中心加速度。
        SGridData<double> internalAcceleration;
        /// 沙内力中法向应力导数和切向应力导数的独立贡献。
        SGridData<double> internalNormalAcceleration;
        SGridData<double> internalTangentialAcceleration;
        /// 内力质量除法使用的原始面高和正则化后的有效面高。
        SGridData<double> internalFaceHeight;
        SGridData<double> internalEffectiveHeight;

        // 每种材料的单元占据掩码，以及由单元掩码扩展得到的面有效性掩码。
        GridData<std::uint8_t> sandActive;
        GridData<std::uint8_t> pureWaterActive;
        GridData<std::uint8_t> mixedWaterActive;
        /// 高度更新前各材料的有效速度面；用于向本步新占据的面外推速度。
        SGridData<std::uint8_t> sandFaceValid;
        SGridData<std::uint8_t> pureWaterFaceValid;
        SGridData<std::uint8_t> mixedWaterFaceValid;
        /// 两层水共享的地形接触开放面。它只描述几何可穿越性，不描述某个
        /// 水层是否占据该面；pure/mixed 必须分别与自己的材料面掩码求交。
        SGridData<std::uint8_t> waterTerrainFaceOpen;
        /// 沙层的地形接触开放面，以真实沙面 H+hs 而非含水载荷势判断。
        SGridData<std::uint8_t> sandTerrainFaceOpen;
        /// 当前子阶段使用的“材料支撑 && 地形开放”组合面掩码。
        SGridData<std::uint8_t> faceValid;

        // M7 本构连通分量遍历缓存；复用可避免每次塑性投影分配 visited/frontier。
        std::vector<std::uint8_t> constitutiveVisited;
        std::vector<Vector2i> constitutiveFrontier;
    };
}
