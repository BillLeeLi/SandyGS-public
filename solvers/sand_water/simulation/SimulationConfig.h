#pragma once

#include "Common.h"

namespace Pivot
{
    /// 水分随沙体演化的模式。
    ///
    /// Dynamic 是论文求解器的完整 pure/mixed 两层水管线。FrozenBound 仅用于
    /// 本构诊断：初始化时位于沙内的水作为守恒示踪量随沙共同平流，不再独立
    /// 求解水动力学，从而隔离“局部饱和度重分布”对沙堆形态的影响。
    enum class SaturationEvolutionMode
    {
        Dynamic,
        FrozenBound
    };

    /// MAC 网格的几何配置。resolution 包含边界带单元。
    struct GridConfig
    {
        /// 单元中心网格在 x/y 方向的总单元数（包括 ghost/boundary cells）。
        Vector2i resolution{64, 48};
        /// 正方形单元边长；本项目的高度和水平坐标使用一致的长度单位。
        double spacing{0.04};
        /// 每侧边界带宽度。目前离散算子至少需要一层边界单元。
        int boundaryWidth{1};
        /// 整个计算域的世界坐标中心。
        Vector2d center{Vector2d::Zero()};
    };

    /// 论文中随含水状态变化的沙体黏聚力模型参数。
    ///
    /// c0/c1/c2 是分段模型使用的黏聚强度，phi1/phi2/phi3 是严格递增的
    /// 饱和度阈值。M3 仅保存并校验这些参数，M4 本构模块会实际使用它们。
    struct CohesionParameters
    {
        double c0{0.0};
        double c1{2000.0};
        double c2{1000.0};
        double phi1{0.20};
        double phi2{0.25};
        double phi3{0.33};
    };

    /// 所有会参与物理方程的材料常数。
    ///
    /// 参数集中存放，避免水、沙和耦合模块各自保存一份而产生不一致。
    struct PhysicalParameters
    {
        /// 默认保持完整动态耦合；FrozenBound 必须由实验配置显式开启。
        SaturationEvolutionMode saturationMode{
            SaturationEvolutionMode::Dynamic};
        /// 重力加速度的大小，模拟默认令重力沿高度负方向。
        double gravity{9.81};
        /// 单位体积沙和水的密度。
        double sandDensity{1600.0};
        double waterDensity{1000.0};
        /// 水在沙层中的显式扩散系数。
        double diffusionRate{1.0e-3};
        /// M5 中沙层与混合水层相对运动的动量交换速率 c_e。
        /// M2 的 pure/mixed 容量转换由 donor 质量直接携带动量，不使用该系数。
        double momentumExchangeRate{1.0};
        /// 沙层与地形接触时使用的摩擦系数。
        double groundFriction{0.5};
        /// 沙前沿越过干地时允许忽略的微小地形高差。
        ///
        /// 该容差只用于沙—地形面的开放判定，不会修改地形、沙高或水相边界。
        /// 默认 0 保留原有严格接触语义；正值可吸收重建地面的微小起伏。
        double sandTerrainContactTolerance{0.0};

        /// 沙体弹性响应的 Lamé 参数。
        double lameMu{3.0e4};
        double lameLambda{3.0e4};
        /// Drucker-Prager 屈服面中的摩擦参数。
        double druckerPragerFriction{0.5};
        CohesionParameters cohesion{};
        /// 诊断开关：false 时跳过 div(h*sigma)，但仍执行重力、平流和摩擦。
        ///
        /// 该开关用于做内力/重力消融实验，不应作为常规模拟的稳定化手段。
        bool enableSandInternalForce{true};
        /// 在 h/theta/P 守恒平流之后，按目的单元的最新含水率再次返回映射 F。
        ///
        /// 默认关闭以保持已有实验的算子顺序。该诊断开关用于检验：平流和
        /// 沙高更新是否会让本构状态相对新的 c(phi) 屈服面重新越界。
        bool projectDeformationAfterAdvection{false};
        /// 仅对本构模型读取的局部饱和度执行一次 3x3 binomial 低通。
        ///
        /// 该诊断开关不会改写 pure/mixed water、sand height 或任何守恒输运
        /// 状态，也不会改变渲染器读取的湿度。它只检验网格尺度含水率扰动经
        /// c(phi)、塑性投影和应力放大后是否是沙面褶皱的主要来源。
        bool constitutiveSaturationLowpass{false};
    };

    /// 时间离散、稳定性阈值和退化状态判定参数。
    struct TimeIntegrationParameters
    {
        /// 一次水更新使用的外层时间步长。
        double waterDt{0.01};
        /// 每个水时间步内执行的沙体子步数。
        int sandSubsteps{3};
        /// 小于该值的材料高度按空单元处理。
        double heightEpsilon{1.0e-8};
        /// 计算沙内力加速度时使用的最小有效面高度。
        ///
        /// 0 表示仅使用 heightEpsilon，完全保留原离散；正值用于避免自由表面
        /// 极薄面把邻近厚沙柱的切向应力散度除以近零质量。它不删除沙质量，
        /// 也不改变平流的 active 判定。
        double internalForceHeightFloor{0.0};
        /// 与网格间距相乘得到分辨率相关的内力面高下限。
        ///
        /// 默认 4e-4 在 dx=0.025 时给出 1e-5；显式绝对下限与该值取最大。
        /// 设为 0 可复现实验中的未正则化离散。
        double internalForceHeightFraction{4.0e-4};
        /// det(F) 必须大于该值，防止沙体单元翻转或奇异。
        double determinantEpsilon{1.0e-8};
        /// 自适应时间步希望满足的最大 CFL 数。
        double targetCfl{0.5};
        /// 显式扩散稳定上限的安全缩放系数。
        double diffusionSafety{0.9};
        /// 数值保护所允许的最大面速度。
        double maxVelocity{10.0};
        /// 为 false 时 Validate() 会直接检查固定时间步的扩散稳定性。
        bool useAdaptiveDt{false};
    };

    /// 沙水动量交换的离散方式。保留论文顺序更新和解析隐式更新两种选择。
    enum class SandWaterExchangeScheme
    {
        PaperSequential,
        AnalyticImplicit
    };

    /// 一次 Simulation 的完整只读配置。
    ///
    /// Simulation 构造时会调用 Validate()，构造成功后各模块共享同一份配置，
    /// 因此运行中不应直接修改这些参数。
    struct SimulationConfig
    {
        GridConfig grid{};
        PhysicalParameters physics{};
        TimeIntegrationParameters time{};
        SandWaterExchangeScheme exchangeScheme{SandWaterExchangeScheme::PaperSequential};

        /// 检查几何、材料参数和固定时间步是否满足基本定义域及稳定性要求。
        /// 参数非法时抛出 std::invalid_argument，并给出可直接定位的原因。
        void Validate() const
        {
            auto const require = [](bool condition, std::string_view message)
            {
                if (!condition)
                {
                    throw std::invalid_argument(std::string(message));
                }
            };

            require((grid.resolution.array() > 0).all(), "grid resolution must be positive");
            require(grid.spacing > 0.0, "grid spacing must be positive");
            require(grid.boundaryWidth >= 1, "boundary width must be at least one cell");
            require((grid.resolution.array() > 2 * grid.boundaryWidth).all(),
                    "grid resolution must contain cells beyond both boundary bands");

            require(physics.gravity >= 0.0, "gravity must be non-negative");
            require(physics.sandDensity > 0.0, "sand density must be positive");
            require(physics.waterDensity > 0.0, "water density must be positive");
            require(physics.diffusionRate >= 0.0, "diffusion rate must be non-negative");
            require(physics.momentumExchangeRate >= 0.0,
                    "momentum exchange rate must be non-negative");
            require(physics.groundFriction >= 0.0, "ground friction must be non-negative");
            require(physics.sandTerrainContactTolerance >= 0.0,
                    "sand terrain contact tolerance must be non-negative");
            require(physics.lameMu > 0.0 && physics.lameLambda >= 0.0,
                    "Lame parameters must define a non-degenerate elastic model");
            require(physics.druckerPragerFriction >= 0.0,
                    "Drucker-Prager friction must be non-negative");
            require(
                physics.saturationMode == SaturationEvolutionMode::Dynamic ||
                    physics.saturationMode == SaturationEvolutionMode::FrozenBound,
                "saturation evolution mode is invalid");

            auto const &cohesion = physics.cohesion;
            require(cohesion.c0 >= 0.0 && cohesion.c1 >= 0.0 && cohesion.c2 >= 0.0,
                    "cohesion values must be non-negative");
            require(0.0 < cohesion.phi1 && cohesion.phi1 < cohesion.phi2 &&
                        cohesion.phi2 < cohesion.phi3 && cohesion.phi3 <= 1.0,
                    "cohesion saturation thresholds must satisfy 0 < phi1 < phi2 < phi3 <= 1");

            require(time.waterDt > 0.0, "water time step must be positive");
            require(time.sandSubsteps >= 1, "sand substeps must be at least one");
            require(time.heightEpsilon > 0.0, "height epsilon must be positive");
            require(time.internalForceHeightFloor >= 0.0,
                    "internal-force height floor must be non-negative");
            require(time.internalForceHeightFraction >= 0.0,
                    "internal-force height fraction must be non-negative");
            require(time.determinantEpsilon > 0.0, "determinant epsilon must be positive");
            require(time.targetCfl > 0.0, "target CFL must be positive");
            require(0.0 < time.diffusionSafety && time.diffusionSafety <= 1.0,
                    "diffusion safety must be in (0, 1]");
            require(time.maxVelocity > 0.0, "maximum velocity must be positive");

            // enum 也可能由配置文件中的整数强制转换得到，不能假定它一定是
            // 两个已声明枚举值之一。
            require(
                exchangeScheme == SandWaterExchangeScheme::PaperSequential ||
                    exchangeScheme == SandWaterExchangeScheme::AnalyticImplicit,
                "sand-water exchange scheme is invalid");

            if (physics.saturationMode == SaturationEvolutionMode::Dynamic &&
                physics.diffusionRate > 0.0)
            {
                // 二维五点 Laplacian 的显式 Euler 稳定条件：
                // dt <= dx^2 / (4 D)，再乘 diffusionSafety 留出安全余量。
                double const diffusionLimit =
                    grid.spacing * grid.spacing / (4.0 * physics.diffusionRate);
                require(time.useAdaptiveDt || time.waterDt <= diffusionLimit * time.diffusionSafety,
                        "fixed water time step violates the explicit diffusion limit");
            }

            if (physics.saturationMode == SaturationEvolutionMode::Dynamic &&
                !time.useAdaptiveDt &&
                exchangeScheme ==
                    SandWaterExchangeScheme::PaperSequential &&
                physics.momentumExchangeRate > 0.0)
            {
                // mixedHeight <= sandHeight 给出阻力特征率的全局上界
                // c_e*(1/rho_s+1/rho_w)。要求单个 sandDt 不越过速度平衡点。
                double const maximumExchangeRate =
                    physics.momentumExchangeRate *
                    (1.0 / physics.sandDensity +
                     1.0 / physics.waterDensity);
                require(
                    time.waterDt /
                            static_cast<double>(time.sandSubsteps) *
                            maximumExchangeRate <=
                        1.0,
                    "fixed sand time step can overshoot explicit momentum exchange");
            }
        }
    };
}
