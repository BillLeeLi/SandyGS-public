#pragma once

#include "diagnostics/Diagnostics.h"
#include "physics/SandSolver.h"
#include "physics/WaterSolver.h"
#include "scene/Scene.h"

namespace Pivot
{
    /// M7 用于无渲染基准的单步阶段耗时。
    ///
    /// 这些时间只用于定位 CPU 热点，不参与物理状态或自适应步长。各字段均为
    /// wall-clock 秒，并在每次 StepWithDt() 开头清零。
    struct SimulationStepTimings
    {
        double sourcesAndSetup{0.0};
        double water{0.0};
        double preSandDiagnostics{0.0};
        double sandDynamics{0.0};
        double layerCorrection{0.0};
        double sandFinalize{0.0};
        double validation{0.0};
        double diagnostics{0.0};
        double total{0.0};
    };

    /// 模拟系统的唯一外部入口，负责按论文顺序组织各个数值模块。
    ///
    /// M6 中 Step() 先施加场景源项、推进两层水，再执行多个沙子步；沙动力学
    /// 内部会与 mixed water 交换动量，并在每个沙子步的动力学与摩擦之间
    /// 修正水层。Solver 只更新自身物理，完整顺序始终由这里管理。
    class Simulation
    {
    public:
        explicit Simulation(SimulationConfig config);

        Simulation(Simulation const &) = delete;
        Simulation &operator=(Simulation const &) = delete;
        Simulation(Simulation &&) = delete;
        Simulation &operator=(Simulation &&) = delete;

        /// 清空状态和缓存，并建立一个满足边界条件的 t=0 空场。
        void Reset();

        /// 用 M6 SceneDescription 初始化高度并接管其中的时间源项。
        ///
        /// SceneDescription 是 move-only；按值传入明确表示其 sources 的所有权
        /// 会转移给 Simulation，调用端应使用 Initialize(std::move(scene))。
        void Initialize(SceneDescription scene);

        /// 从三个解析高度函数初始化地形、沙和“总水量”。
        ///
        /// 水优先填入沙层内部：mixed=min(totalWater,sand)，其余成为 pureWater。
        /// 负材料高度会被截断为零，非有限返回值会立即报错。
        void InitializeHeights(
            HeightFunction const &terrain,
            HeightFunction const &sandHeight,
            HeightFunction const &totalWaterHeight);

        /// 推进一个水大步：混合水扩散、两层平流、转换、压力与速度更新。
        void Step();
        /// 推进一个水大步，但允许调用者为本步给出更小的上限。
        ///
        /// 长程 duration 模式用它缩短最后一步以精确命中目标物理时间；普通
        /// CFL、自适应扩散和弹性波速限制仍会在该上限内生效。
        void Step(double maximumDt);
        /// 重复执行水步直到到达 targetTime；最后一步会缩短以精确命中目标时间。
        void AdvanceTo(double targetTime);

        /// 计算当前状态的守恒量与稳定性指标，不修改模拟状态。
        ConservationReport Measure() const;
        /// 在指定阶段执行有限值、非负性、本构和边界约束检查。
        void Validate(std::string_view stage) const;

        // 只读访问配置/网格；状态提供可写版本以便后续数值模块更新。
        SimulationConfig const &GetConfig() const { return m_Config; }
        StaggeredGrid const &GetGrid() const { return m_Grid; }
        SimulationState const &GetState() const { return m_State; }
        SimulationState &GetState() { return m_State; }
        SimulationScratch const &GetScratch() const { return m_Scratch; }
        ConservationReport const &GetLastReport() const { return m_LastReport; }
        WaterStepReport const &GetLastWaterStepReport() const
        {
            return m_LastWaterStepReport;
        }
        SandStepReport const &GetLastSandStepReport() const
        {
            return m_LastSandStepReport;
        }
        /// 返回最近一步各阶段的 wall-clock 耗时；仅供 M7 benchmark/profiler。
        SimulationStepTimings const &GetLastStepTimings() const
        {
            return m_LastStepTimings;
        }

    private:
        /// 根据目标步长、浅水特征波速、速度 CFL 和扩散上限选择安全时间步。
        double ComputeWaterDt(double maximumDt) const;
        /// 只积分沙高；M7 用它替代包含本构 SVD 的完整 Measure() 预统计。
        double ComputeSandVolume() const;
        void StepWithDt(double dt);
        /// 在水步开始前施加与 [beginTime,endTime] 相交的所有 M6 源项。
        void ApplySources(double beginTime, double endTime);
        /// 检查 FrozenBound 的定义域：无自由水、无源项，且混合水不超沙容量。
        void ValidateFrozenBoundState(std::string_view stage) const;
        /// 令束缚水面速度与完成摩擦/限幅后的沙速度完全一致。
        void SynchronizeFrozenBoundWaterVelocity();
        /// 任何会改变高度或速度的阶段结束后调用：填充边界并重建材料掩码。
        void ApplyBoundariesAndRefreshMasks();

        // 成员声明顺序也是构造依赖顺序：配置 -> 网格 -> 网格数据 -> 服务模块。
        SimulationConfig m_Config;
        StaggeredGrid m_Grid;
        SimulationState m_State;
        SimulationScratch m_Scratch;
        BoundaryConditions m_Boundary;
        WaterSolver m_WaterSolver;
        SandSolver m_SandSolver;
        Diagnostics m_Diagnostics;
        ConservationReport m_LastReport;
        WaterStepReport m_LastWaterStepReport;
        SandStepReport m_LastSandStepReport;
        SimulationStepTimings m_LastStepTimings;
        /// Simulation 独占源项，保证多步运行中其生命周期稳定。
        std::vector<std::unique_ptr<Source>> m_Sources;
    };
}
