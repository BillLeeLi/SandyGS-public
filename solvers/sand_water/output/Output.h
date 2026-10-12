#pragma once

#include "diagnostics/Diagnostics.h"

namespace Pivot
{
    /// 控制每帧导出的数据量。
    enum class SnapshotLevel
    {
        /// 仅保存构造三维高度表面所需的地形和材料高度。
        ///
        /// 长程 M6.5 运行使用该级别控制磁盘占用；守恒/稳定性标量仍会写入
        /// metrics.csv，因此它不是关闭诊断，只是不保存逐单元速度与本构场。
        Render,
        /// 高度、饱和度和各材料速度，足够生成常规概览。
        Basic,
        /// Basic 加上 F、det(F)、旋转角和拉伸矩阵等本构诊断量。
        Diagnostic
    };

    /// 提交给输出模块的轻量只读快照视图。
    ///
    /// 该类型不复制大型网格数据，因此 state 的生命周期必须覆盖 WriteFrame()。
    /// time/stepIndex 在构造时单独捕获，避免输出命名与状态标签不一致。
    struct SimulationSnapshot
    {
        explicit SimulationSnapshot(SimulationState const &simulationState)
            : state(simulationState),
              scratch(nullptr),
              time(simulationState.time),
              stepIndex(simulationState.stepIndex)
        {
        }

        /// 同时捕获工作缓存时，输出模块会额外写出最近一步的层转换高度。
        SimulationSnapshot(
            SimulationState const &simulationState,
            SimulationScratch const &simulationScratch)
            : state(simulationState),
              scratch(&simulationScratch),
              time(simulationState.time),
              stepIndex(simulationState.stepIndex)
        {
        }

        SimulationState const &state;
        SimulationScratch const *scratch;
        double time;
        std::uint64_t stepIndex;
    };

    /// 将模拟状态写成可由 Python/NumPy 直接读取的运行目录。
    ///
    /// 目录协议：
    ///   metadata.yaml       固定配置和格式版本
    ///   metrics.csv         每帧一行的守恒/稳定性时间序列
    ///   frames/000000/*.npy 单帧网格数据
    ///   previews/           可视化脚本的默认输出位置
    class FrameExporter
    {
    public:
        /// 创建运行目录，写入元数据，并重新创建 metrics.csv 表头。
        /// 因此每个 run 只能在开始时调用一次。
        void WriteMetadata(
            SimulationConfig const &config,
            std::filesystem::path const &runDirectory,
            std::string_view sceneName = {},
            SnapshotLevel level = SnapshotLevel::Basic) const;

        /// 仅追加一行标量历史，不写任何网格数组。
        ///
        /// M6.5 长程运行每步调用它，使短暂的 CFL、速度钳制或 det(F) 极值
        /// 不会因 frameStride 较大而从审计历史中消失。
        void WriteMetrics(
            SimulationSnapshot const &snapshot,
            ConservationReport const &report,
            std::filesystem::path const &runDirectory) const;

        /// 按 stepIndex 写入一个帧目录；appendMetrics 保留独立调用时的兼容行为。
        /// 返回实际帧目录，便于调用端记录或继续处理。
        std::filesystem::path WriteFrame(
            SimulationSnapshot const &snapshot,
            ConservationReport const &report,
            SimulationConfig const &config,
            SnapshotLevel level,
            std::filesystem::path const &runDirectory,
            bool appendMetrics = true) const;

        /// 将当前帧导出为三张共享规则拓扑的 OBJ 高度表面。
        ///
        /// 输出目录中生成 terrain.obj、sand_surface.obj 和
        /// water_surface.obj；只导出物理内部域，不包含 ghost/boundary cells。
        void WriteObjMeshes(
            SimulationSnapshot const &snapshot,
            SimulationConfig const &config,
            std::filesystem::path const &directory) const;

    private:
        /// FrameExporter 内部使用的最小三角网格表示；faces 保存零基索引。
        struct HeightMesh
        {
            std::vector<Vector3d> vertices;
            std::vector<Vector3i> faces;
        };

        /// 构造 z=terrain+firstHeight+secondHeight 的规则网格。
        static HeightMesh BuildHeightMesh(
            GridData<double> const &terrain,
            GridData<double> const *firstHeight,
            GridData<double> const *secondHeight,
            int boundaryWidth);

        static void WriteObj(
            std::filesystem::path const &path,
            HeightMesh const &mesh);
    };
}
