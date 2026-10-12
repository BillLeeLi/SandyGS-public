#include "simulation/Simulation.h"

namespace Pivot
{
    namespace
    {
        using StepClock = std::chrono::steady_clock;

        double ElapsedSeconds(
            StepClock::time_point begin,
            StepClock::time_point end)
        {
            return std::chrono::duration<double>(end - begin).count();
        }

        // 成员按声明顺序构造，而 StaggeredGrid 依赖合法配置。使用这个辅助函数
        // 可确保配置在进入 m_Grid 构造函数之前完成校验。
        SimulationConfig ValidateConfig(SimulationConfig config)
        {
            config.Validate();
            return config;
        }
    }

    Simulation::Simulation(SimulationConfig config)
        : m_Config(ValidateConfig(std::move(config))),
          m_Grid(
              m_Config.grid.boundaryWidth,
              m_Config.grid.spacing,
              m_Config.grid.resolution,
              m_Config.grid.center),
          m_State(m_Grid),
          m_Scratch(m_Grid),
          m_Boundary(m_Grid)
    {
        Reset();
    }

    void Simulation::Reset()
    {
        m_Sources.clear();
        m_State.SetZero();
        m_Scratch.SetZero();
        ApplyBoundariesAndRefreshMasks();
        Validate("reset");
        m_LastReport = Measure();
        m_LastWaterStepReport = {};
        m_LastSandStepReport = {};
    }

    void Simulation::Initialize(SceneDescription scene)
    {
        // InitializeHeights 会清空旧源项；先暂存新所有权，完成高度初始化后
        // 再提交，避免重置过程误删刚从场景中移动出的对象。
        std::vector<std::unique_ptr<Source>> sources =
            std::move(scene.sources);
        if (m_Config.physics.saturationMode ==
                SaturationEvolutionMode::FrozenBound &&
            !sources.empty())
        {
            throw std::invalid_argument(
                "frozen_bound saturation mode does not support time-dependent sources");
        }
        InitializeHeights(
            scene.terrain,
            scene.sandHeight,
            scene.totalWaterHeight);
        m_Sources = std::move(sources);
    }

    void Simulation::InitializeHeights(
        HeightFunction const &terrain,
        HeightFunction const &sandHeight,
        HeightFunction const &totalWaterHeight)
    {
        m_Sources.clear();
        m_State.SetZero();
        ParallelForEach(m_Grid.GetCellGrid(), [&](Vector2i const &cell)
                        {
            Vector2d const position = m_Grid.GetCellGrid().PositionOf(cell);
            double const H = terrain(position);
            double const hs = std::max(sandHeight(position), 0.0);
            double const totalWater = std::max(totalWaterHeight(position), 0.0);

            if (!std::isfinite(H) || !std::isfinite(hs) || !std::isfinite(totalWater))
            {
                throw std::runtime_error(fmt::format(
                    "initializer returned a non-finite value at ({}, {})",
                    cell.x(), cell.y()));
            }

            m_State.terrain[cell] = H;
            m_State.sand.height[cell] = hs;
            // 高度场模型把水分成沙内混合水和沙上自由水。初始化采用论文的
            // 容量约束：沙柱最多容纳与自身高度相同的混合水。
            m_State.mixedWater.height[cell] = std::min(totalWater, hs);
            m_State.pureWater.height[cell] =
                totalWater - m_State.mixedWater.height[cell]; });

        ApplyBoundariesAndRefreshMasks();
        if (m_Config.physics.saturationMode ==
            SaturationEvolutionMode::FrozenBound)
        {
            ValidateFrozenBoundState("initialize heights");
            SynchronizeFrozenBoundWaterVelocity();
        }
        Validate("initialize heights");
        m_LastReport = Measure();
        m_LastWaterStepReport = {};
        m_LastSandStepReport = {};
    }

    double Simulation::ComputeWaterDt(double maximumDt) const
    {
        double dt = std::min(m_Config.time.waterDt, maximumDt);
        if (!m_Config.time.useAdaptiveDt)
        {
            return dt;
        }

        if (m_Config.physics.saturationMode == SaturationEvolutionMode::Dynamic)
        {
            // 对浅水方程，信息传播速度不仅包含流速，还包含重力波速 sqrt(g*h)。
            // FrozenBound 不推进浅水方程，因此不应让该虚构波速缩短诊断步长。
            double maxHeight = 0.0;
            ForEach(m_Grid.GetCellGrid(), [&](Vector2i const &cell)
            {
                if (m_Grid.IsInsideCell(cell))
                {
                    maxHeight = std::max(
                        maxHeight,
                        m_State.pureWater.height[cell] +
                            m_State.mixedWater.height[cell]);
                }
            });
            double const maxVelocity = std::max(
                m_State.pureWater.velocity.GetMaxAbsComponent(),
                m_State.mixedWater.velocity.GetMaxAbsComponent());
            double const characteristicSpeed =
                maxVelocity + std::sqrt(m_Config.physics.gravity * maxHeight);
            if (characteristicSpeed > m_Config.time.heightEpsilon)
            {
                dt = std::min(
                    dt,
                    m_Config.time.targetCfl *
                        m_Grid.GetSpacing() / characteristicSpeed);
            }
        }

        // 沙在一个外层水步中执行 T 个子步，因此外层 dt 的沙 CFL 上限可放宽
        // T 倍。特征高度同时计入纯水柱对沙施加的密度缩放载荷。
        double maxSandPotentialHeight = 0.0;
        ForEach(m_Grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (m_Grid.IsInsideCell(cell))
            {
                maxSandPotentialHeight = std::max(
                    maxSandPotentialHeight,
                    m_State.sand.height[cell] +
                        m_Config.physics.waterDensity /
                            m_Config.physics.sandDensity *
                            m_State.pureWater.height[cell]);
            }
        });
        double const maxSandVelocity =
            m_State.sand.velocity.GetMaxAbsComponent();
        double const sandCharacteristicSpeed =
            maxSandVelocity +
            std::sqrt(
                m_Config.physics.gravity *
                maxSandPotentialHeight);
        if (sandCharacteristicSpeed > m_Config.time.heightEpsilon)
        {
            dt = std::min(
                dt,
                m_Config.time.targetCfl *
                    static_cast<double>(m_Config.time.sandSubsteps) *
                    m_Grid.GetSpacing() /
                    sandCharacteristicSpeed);
        }

        // 内力采用显式更新，稳定步长还必须覆盖纵向弹性波速。
        // 外层一步包含 sandSubsteps 个沙步，故上限相应乘以子步数。
        if (m_Config.physics.enableSandInternalForce)
        {
            double const elasticWaveSpeed = std::sqrt(
                (m_Config.physics.lameLambda +
                 2.0 * m_Config.physics.lameMu) /
                m_Config.physics.sandDensity);
            dt = std::min(
                dt,
                m_Config.time.targetCfl *
                    static_cast<double>(m_Config.time.sandSubsteps) *
                    m_Grid.GetSpacing() /
                    elasticWaveSpeed);
        }

        // 论文顺序模式是显式阻力更新。由 mixedHeight<=sandHeight 可得
        // lambda <= c_e(1/rho_s+1/rho_w)；令 sandDt*lambda<=targetCfl
        // 可防止相对速度在一次常规 Simulation 子步内跨过零点。
        if (m_Config.physics.saturationMode == SaturationEvolutionMode::Dynamic &&
            m_Config.exchangeScheme ==
                SandWaterExchangeScheme::PaperSequential &&
            m_Config.physics.momentumExchangeRate > 0.0)
        {
            double const maximumExchangeRate =
                m_Config.physics.momentumExchangeRate *
                (1.0 / m_Config.physics.sandDensity +
                 1.0 / m_Config.physics.waterDensity);
            dt = std::min(
                dt,
                m_Config.time.targetCfl *
                    static_cast<double>(m_Config.time.sandSubsteps) /
                    maximumExchangeRate);
        }

        // M2 的显式五点扩散必须满足 dt <= dx^2/(4D)。
        if (m_Config.physics.saturationMode == SaturationEvolutionMode::Dynamic &&
            m_Config.physics.diffusionRate > 0.0)
        {
            double const diffusionDt =
                m_Config.time.diffusionSafety *
                m_Grid.GetSpacing() * m_Grid.GetSpacing() /
                (4.0 * m_Config.physics.diffusionRate);
            dt = std::min(dt, diffusionDt);
        }
        return dt;
    }

    void Simulation::Step()
    {
        Step(std::numeric_limits<double>::infinity());
    }

    void Simulation::Step(double maximumDt)
    {
        if (!(maximumDt > 0.0) || std::isnan(maximumDt))
        {
            throw std::invalid_argument(
                "maximum time step must be positive and not NaN");
        }
        StepWithDt(ComputeWaterDt(maximumDt));
    }

    void Simulation::AdvanceTo(double targetTime)
    {
        if (!std::isfinite(targetTime) || targetTime < m_State.time)
        {
            throw std::invalid_argument(
                "target time must be finite and not earlier than current time");
        }

        double const timeTolerance =
            16.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, std::abs(targetTime));
        while (targetTime - m_State.time > timeTolerance)
        {
            double const remaining = targetTime - m_State.time;
            double const dt = ComputeWaterDt(remaining);
            if (!(dt > 0.0))
            {
                throw std::runtime_error("time-step controller produced a non-positive dt");
            }
            StepWithDt(dt);
        }
        // 消除连续加法产生的舍入级尾差，使调用者能够可靠比较目标时间标签。
        m_State.time = targetTime;
    }

    double Simulation::ComputeSandVolume() const
    {
        double volume = 0.0;
        double const area =
            m_Grid.GetSpacing() * m_Grid.GetSpacing();
        ForEach(m_Grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (m_Grid.IsInsideCell(cell))
            {
                volume += m_State.sand.height[cell] * area;
            }
        });
        return volume;
    }

    void Simulation::StepWithDt(double dt)
    {
        m_LastStepTimings = {};
        StepClock::time_point const totalBegin = StepClock::now();

        ApplySources(m_State.time, m_State.time + dt);
        StepClock::time_point stageBegin = StepClock::now();
        m_LastStepTimings.sourcesAndSetup =
            ElapsedSeconds(totalBegin, stageBegin);

        if (m_Config.physics.saturationMode == SaturationEvolutionMode::Dynamic)
        {
            m_LastWaterStepReport = m_WaterSolver.Step(
                m_State,
                m_Config,
                dt,
                m_Boundary,
                m_Scratch);
        }
        else
        {
            // FrozenBound 是隔离饱和度重分布的消融，不执行独立浅水、扩散或
            // pure/mixed 转换；水仍会在后面的沙子平流中作为束缚示踪量移动。
            m_LastWaterStepReport = {};
        }
        StepClock::time_point stageEnd = StepClock::now();
        m_LastStepTimings.water =
            ElapsedSeconds(stageBegin, stageEnd);

        m_LastSandStepReport = {};
        stageBegin = stageEnd;
        m_LastSandStepReport.volumeBefore =
            ComputeSandVolume();
        stageEnd = StepClock::now();
        m_LastStepTimings.preSandDiagnostics =
            ElapsedSeconds(stageBegin, stageEnd);
        m_LastSandStepReport.volumeAfterStep =
            m_LastSandStepReport.volumeBefore;
        m_LastSandStepReport.minHeight =
            std::numeric_limits<double>::infinity();

        int const substepCount = m_Config.time.sandSubsteps;
        double const sandDt = dt / static_cast<double>(substepCount);
        // 空沙场无需执行沙循环。无重力/内力且沙静止时通常也可跳过，但 M5
        // 必须保留“沙静止、混合水运动”的交换情形，否则阻力永远不会启动。
        double const minimumMaterialVolume =
            m_Config.time.heightEpsilon *
            m_Grid.GetSpacing() *
            m_Grid.GetSpacing();
        bool const hasMovingMixedWaterForExchange =
            m_Config.physics.saturationMode == SaturationEvolutionMode::Dynamic &&
            m_Config.physics.momentumExchangeRate > 0.0 &&
            m_State.mixedWater.velocity.GetMaxAbsComponent() >
                m_Config.time.heightEpsilon;
        bool const requiresSandUpdate =
            m_LastSandStepReport.volumeBefore > minimumMaterialVolume &&
            (m_Config.physics.gravity > 0.0 ||
             m_Config.physics.enableSandInternalForce ||
             m_State.sand.velocity.GetMaxAbsComponent() >
                m_Config.time.heightEpsilon ||
             hasMovingMixedWaterForExchange);
        for (int substep = 0;
             substep < (requiresSandUpdate ? substepCount : 0);
             ++substep)
        {
            stageBegin = StepClock::now();
            m_SandSolver.IntegrateDynamics(
                m_State,
                m_Config,
                sandDt,
                m_Boundary,
                m_Scratch,
                m_LastSandStepReport,
                substep == 0);
            stageEnd = StepClock::now();
            m_LastStepTimings.sandDynamics +=
                ElapsedSeconds(stageBegin, stageEnd);

            stageBegin = stageEnd;
            WaterStepReport const correction =
                m_Config.physics.saturationMode == SaturationEvolutionMode::Dynamic
                ? m_WaterSolver.CorrectLayersAfterSandMotion(
                    m_State,
                    m_Config,
                    sandDt,
                    m_Boundary,
                    m_Scratch)
                : WaterStepReport{};
            stageEnd = StepClock::now();
            m_LastStepTimings.layerCorrection +=
                ElapsedSeconds(stageBegin, stageEnd);
            m_LastWaterStepReport.maxAbsLayerTransfer = std::max(
                m_LastWaterStepReport.maxAbsLayerTransfer,
                correction.maxAbsLayerTransfer);
            m_LastWaterStepReport.maxLayerMomentumError = std::max(
                m_LastWaterStepReport.maxLayerMomentumError,
                correction.maxLayerMomentumError);
            m_LastWaterStepReport.totalVolumeAfterStep =
                correction.totalVolumeAfterStep;

            stageBegin = stageEnd;
            m_SandSolver.ApplyFrictionAndFinalize(
                m_State,
                m_Config,
                sandDt,
                m_Boundary,
                m_Scratch,
                m_LastSandStepReport);
            if (m_Config.physics.saturationMode ==
                SaturationEvolutionMode::FrozenBound)
            {
                // 复制最终提交的沙速度，而不是摩擦前候选速度。
                SynchronizeFrozenBoundWaterVelocity();
                ValidateFrozenBoundState("frozen-bound sand substep");
            }
            stageEnd = StepClock::now();
            m_LastStepTimings.sandFinalize +=
                ElapsedSeconds(stageBegin, stageEnd);

        }
        if (!std::isfinite(m_LastSandStepReport.minHeight))
        {
            m_LastSandStepReport.minHeight = 0.0;
        }

        m_State.lastDt = dt;
        m_State.time += dt;
        ++m_State.stepIndex;
        stageBegin = StepClock::now();
        ApplyBoundariesAndRefreshMasks();
        Validate("end of M6 full step");
        stageEnd = StepClock::now();
        m_LastStepTimings.validation =
            ElapsedSeconds(stageBegin, stageEnd);
        stageBegin = stageEnd;
        m_LastReport = Measure();
        stageEnd = StepClock::now();
        m_LastStepTimings.diagnostics =
            ElapsedSeconds(stageBegin, stageEnd);
        // Measure() 负责从状态计算全局量；这些字段来自刚结束的水子算子，
        // 在此合并可让 metrics.csv 同时携带全局和局部守恒诊断。
        m_LastReport.mixedDiffusionVolumeChange =
            m_LastWaterStepReport.DiffusionVolumeChange();
        m_LastReport.waterAdvectionVolumeChange =
            m_LastWaterStepReport.AdvectionVolumeChange();
        m_LastReport.layerCorrectionVolumeChange =
            m_LastWaterStepReport.LayerCorrectionVolumeChange();
        m_LastReport.maxLayerTransfer =
            m_LastWaterStepReport.maxAbsLayerTransfer;
        m_LastReport.maxLayerMomentumError =
            m_LastWaterStepReport.maxLayerMomentumError;
        m_LastReport.sandAdvectionVolumeChange =
            m_LastSandStepReport.cumulativeAdvectionVolumeChange;
        m_LastReport.maxSandGravityAcceleration =
            m_LastSandStepReport.maxGravityAcceleration;
        m_LastReport.maxExchangeRelativeSpeedBefore =
            m_LastSandStepReport.maxExchangeRelativeSpeedBefore;
        m_LastReport.maxExchangeRelativeSpeedAfter =
            m_LastSandStepReport.maxExchangeRelativeSpeedAfter;
        m_LastReport.maxExchangeImpulse =
            m_LastSandStepReport.maxExchangeImpulse;
        m_LastReport.maxExchangeMomentumError =
            m_LastSandStepReport.maxExchangeMomentumError;
        m_LastReport.exchangedFaceCount =
            m_LastSandStepReport.exchangedFaceCount;
        m_LastReport.exchangeOvershootFaceCount =
            m_LastSandStepReport.exchangeOvershootFaceCount;
        m_LastReport.maxSandInternalAcceleration =
            m_LastSandStepReport.maxInternalAcceleration;
        m_LastReport.sandHeightAtMaxInternalAcceleration =
            std::isfinite(
                m_LastSandStepReport.heightAtMaxInternalAcceleration)
            ? m_LastSandStepReport.heightAtMaxInternalAcceleration
            : 0.0;
        m_LastReport.maxSandNormalStressAcceleration =
            m_LastSandStepReport.maxNormalStressAcceleration;
        m_LastReport.maxSandTangentialStressAcceleration =
            m_LastSandStepReport.maxTangentialStressAcceleration;
        m_LastReport.maxPostAdvectionYieldResidualBefore =
            m_LastSandStepReport.maxPostAdvectionYieldResidualBefore;
        m_LastReport.maxPostAdvectionYieldResidualAfter =
            m_LastSandStepReport.maxPostAdvectionYieldResidualAfter;
        m_LastReport.maxPostAdvectionProjectionDelta =
            m_LastSandStepReport.maxPostAdvectionProjectionDelta;
        m_LastReport.postAdvectionProjectionCount =
            m_LastSandStepReport.postAdvectionProjectionCount;
        m_LastReport.maxSandVelocityBeforeClamping =
            m_LastSandStepReport.maxVelocityBeforeClamping;
        m_LastReport.sandVelocityClampCount =
            m_LastSandStepReport.velocityClampCount;
        m_LastReport.maxSandFrictionDeceleration =
            m_LastSandStepReport.maxFrictionDeceleration;
        m_LastReport.frictionStoppedFaces =
            m_LastSandStepReport.frictionStoppedFaces;
        m_LastStepTimings.total =
            ElapsedSeconds(totalBegin, StepClock::now());
    }

    void Simulation::ValidateFrozenBoundState(std::string_view stage) const
    {
        if (!m_Sources.empty())
        {
            throw std::runtime_error(fmt::format(
                "{}: frozen_bound saturation mode cannot contain sources", stage));
        }
        double const tolerance = 64.0 * std::numeric_limits<double>::epsilon();
        ForEach(m_Grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (!m_Grid.IsInsideCell(cell))
            {
                return;
            }
            double const pure = m_State.pureWater.height[cell];
            double const mixed = m_State.mixedWater.height[cell];
            double const sand = m_State.sand.height[cell];
            if (pure > m_Config.time.heightEpsilon ||
                mixed > sand + tolerance * std::max(1.0, sand))
            {
                throw std::runtime_error(fmt::format(
                    "{}: frozen_bound requires pure_water=0 and mixed_water<=sand "
                    "at ({}, {}), got pure={}, mixed={}, sand={}",
                    stage, cell.x(), cell.y(), pure, mixed, sand));
            }
        });
    }

    void Simulation::SynchronizeFrozenBoundWaterVelocity()
    {
        m_State.pureWater.SetZero();
        m_State.mixedWater.velocity = m_State.sand.velocity;
        m_Boundary.ApplyVelocityBoundary(m_State.mixedWater.velocity);
    }

    void Simulation::ApplySources(
        double beginTime,
        double endTime)
    {
        if (m_Sources.empty())
        {
            return;
        }
        for (std::unique_ptr<Source> const &source : m_Sources)
        {
            source->Apply(m_State, beginTime, endTime);
        }

        // 源项可同时修改中心高度和 MAC 速度；在任一 Solver 读取状态前统一
        // 填充 ghost、恢复封闭边界并重建 active masks。
        ApplyBoundariesAndRefreshMasks();
        Validate("after M6 sources");
    }

    void Simulation::ApplyBoundariesAndRefreshMasks()
    {
        // 顺序很重要：掩码必须依据已经填充好边界带的最新高度生成，
        // 速度诊断则要求掩码刷新前后都满足封闭边界。
        m_Boundary.ApplyHeightBoundary(m_State.terrain);
        m_Boundary.ApplyHeightBoundary(m_State.sand.height);
        m_Boundary.ApplyHeightBoundary(m_State.pureWater.height);
        m_Boundary.ApplyHeightBoundary(m_State.mixedWater.height);
        m_Boundary.ApplyVelocityBoundary(m_State.sand.velocity);
        m_Boundary.ApplyVelocityBoundary(m_State.pureWater.velocity);
        m_Boundary.ApplyVelocityBoundary(m_State.mixedWater.velocity);
        m_Boundary.UpdateMasks(m_State, m_Config.time.heightEpsilon, m_Scratch);
    }

    ConservationReport Simulation::Measure() const
    {
        return m_Diagnostics.Measure(m_State, m_Config, m_Grid);
    }

    void Simulation::Validate(std::string_view stage) const
    {
        m_Diagnostics.Validate(m_State, m_Config, m_Grid, m_Boundary, stage);
    }
}
