#pragma once

#include "simulation/SimulationConfig.h"
#include "simulation/SimulationState.h"

namespace Pivot
{
    /// 根据二维世界坐标返回一个初始高度。
    using HeightFunction = std::function<double(Vector2d const &)>;

    /// M6 批量运行所需的非物理设置；它们不进入数值方程。
    struct SceneRunSettings
    {
        /// 场景名用于日志和 metadata，不参与输出目录推导。
        std::string name{"unnamed"};
        /// 固定步数模式的外层时间步数量。duration>=0 时该值不参与停止判断。
        int steps{100};
        /// 按物理时间运行的终点；负值表示仍使用 steps。
        ///
        /// duration 模式会缩短最后一个时间步，从而精确命中该时刻，适合比较
        /// 不同 CFL 或分辨率下的长程结果。
        double duration{-1.0};
        int frameStride{10};
        /// true 时输出 F、应力与屈服残差等较大的诊断数组。
        bool diagnosticSnapshots{false};
        /// true 时为每个保存帧额外导出 terrain/sand/water 三张 OBJ。
        bool writeObj{false};
    };

    /// 源项共用的二维解析区域。
    ///
    /// Box 使用闭区间 [minimum, maximum]；Circle 使用 center/radius。
    /// 保留为值类型可让三个简单 Source 共享区域逻辑而无需再建立形状继承树。
    struct SourceRegion
    {
        enum class Shape
        {
            Box,
            Circle
        };

        Shape shape{Shape::Box};
        Vector2d minimum{Vector2d::Zero()};
        Vector2d maximum{Vector2d::Zero()};
        Vector2d center{Vector2d::Zero()};
        double radius{0.0};

        /// 判断单元中心或 MAC 面中心是否位于源区域。
        bool Contains(Vector2d const &position) const;
        /// 在 SceneLoader 构造后检查区域是否非退化且参数有限。
        void Validate() const;
    };

    /// 在一个时间区间内修改状态的源项接口。
    class Source
    {
    public:
        virtual ~Source() = default;

        /// 将 [beginTime,endTime] 与源的有效区间求交后施加增量或约束。
        virtual void Apply(
            SimulationState &state,
            double beginTime,
            double endTime) const = 0;
    };

    /// 以 heightRate（高度/秒）向区域内注入总水，并规定入口速度。
    class WaterInlet final : public Source
    {
    public:
        WaterInlet(
            SourceRegion region,
            double heightRate,
            Vector2d velocity,
            double beginTime,
            double endTime);

        void Apply(
            SimulationState &state,
            double beginTime,
            double endTime) const override;

    private:
        SourceRegion region_;
        double heightRate_;
        Vector2d velocity_;
        double beginTime_;
        double endTime_;
    };

    /// 以 heightRate 增加沙高，并给新加入的沙设置规定速度。
    class SandEmitter final : public Source
    {
    public:
        SandEmitter(
            SourceRegion region,
            double heightRate,
            Vector2d velocity,
            double beginTime,
            double endTime);

        void Apply(
            SimulationState &state,
            double beginTime,
            double endTime) const override;

    private:
        SourceRegion region_;
        double heightRate_;
        Vector2d velocity_;
        double beginTime_;
        double endTime_;
    };

    /// VelocityRegion 可约束一种或多种材料的面速度，用于 flush/seepage 入口。
    enum class SourceMaterial
    {
        Sand,
        PureWater,
        MixedWater,
        AllWater,
        All
    };

    class VelocityRegion final : public Source
    {
    public:
        VelocityRegion(
            SourceRegion region,
            SourceMaterial material,
            Vector2d velocity,
            double beginTime,
            double endTime);

        void Apply(
            SimulationState &state,
            double beginTime,
            double endTime) const override;

    private:
        SourceRegion region_;
        SourceMaterial material_;
        Vector2d velocity_;
        double beginTime_;
        double endTime_;
    };

    /// 一个可直接交给 Simulation 的完整场景描述。
    ///
    /// 初始输入只描述总水高；Simulation 会统一执行 mixed=min(water,sand)
    /// 的层划分，因此 YAML 无法构造互相矛盾的 pure/mixed 初值。
    struct SceneDescription
    {
        SimulationConfig config{};
        SceneRunSettings run{};
        HeightFunction terrain;
        HeightFunction sandHeight;
        HeightFunction totalWaterHeight;
        std::vector<std::unique_ptr<Source>> sources;

        SceneDescription();
        SceneDescription(SceneDescription const &) = delete;
        SceneDescription &operator=(SceneDescription const &) = delete;
        SceneDescription(SceneDescription &&) = default;
        SceneDescription &operator=(SceneDescription &&) = default;
    };

    /// 将 YAML 中的网格、物理参数、解析几何、NPY 高度图和源项构造成场景。
    class SceneLoader
    {
    public:
        static SceneDescription FromYaml(
            std::filesystem::path const &path);
    };
}
