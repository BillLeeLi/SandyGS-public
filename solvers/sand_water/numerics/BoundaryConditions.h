#pragma once

#include "simulation/SimulationState.h"

namespace Pivot
{
    /// 矩形计算域边界上的速度条件。
    enum class DomainBoundaryType
    {
        /// 法向速度为零，切向速度作零法向导数延拓。
        ClosedFreeSlip,
        /// 所有边界面速度均为零。
        ClosedNoSlip
    };

    /// 集中处理 ghost cell、MAC 面速度和材料有效性掩码。
    ///
    /// 将边界逻辑放在同一类型中可以保证平流、扩散和投影等模块使用一致的
    /// 封闭域定义。对象仅引用网格，不拥有任何模拟数据。
    class BoundaryConditions
    {
    public:
        explicit BoundaryConditions(
            StaggeredGrid const &grid,
            DomainBoundaryType type = DomainBoundaryType::ClosedFreeSlip);

        /// 依据材料高度重建单元 active mask，并从沙掩码生成默认 faceValid。
        /// 边界带永远标记为空；height <= heightEpsilon 也视作空单元。
        void UpdateMasks(
            SimulationState const &state,
            double heightEpsilon,
            SimulationScratch &scratch) const;

        /// 若一个面至少邻接一个 active cell，则将对应面标记为有效。
        void BuildFaceMask(
            GridData<std::uint8_t> const &cellMask,
            SGridData<std::uint8_t> &faceMask) const;

        /// 构造两层水共享的内部地形接触掩码。若一侧总水深为零且该侧地形
        /// 高于另一侧自由水面，则该面关闭；低位干地仍保持开放以允许润湿。
        void BuildWaterTerrainOpenMask(
            SimulationState const &state,
            double heightEpsilon,
            SGridData<std::uint8_t> &faceOpen) const;

        /// 构造沙—地形接触掩码。仅当无沙侧地形高于有沙侧真实表面 H+hs
        /// 与 contactTolerance 之和时关闭；纯水载荷不属于几何高度，因此不
        /// 参与越障判定。容差只吸收微小重建起伏，不改变任何高度场。
        void BuildSandTerrainOpenMask(
            SimulationState const &state,
            double heightEpsilon,
            double contactTolerance,
            SGridData<std::uint8_t> &faceOpen) const;

        /// 逐面求交材料支撑掩码和接触开放掩码。允许 destination 与任一输入
        /// 为同一对象，便于在预分配缓存中原地组合。
        void IntersectFaceMasks(
            SGridData<std::uint8_t> const &materialFaceValid,
            SGridData<std::uint8_t> const &terrainFaceOpen,
            SGridData<std::uint8_t> &destination) const;

        /// 清零内部地形阻挡面的法向速度。域外边界仍由 ApplyVelocityBoundary
        /// 单独处理，使“场景地形接触”和“矩形计算域边界”保持独立语义。
        void ApplyTerrainContactBoundary(
            SGridData<double> &velocity,
            SGridData<std::uint8_t> const &terrainFaceOpen) const;

        /// 用最近内部单元填充边界带，即标量场的零法向梯度条件。
        void ApplyHeightBoundary(GridData<double> &field) const;
        /// 根据 GetType() 对 MAC 面速度施加封闭域条件。
        void ApplyVelocityBoundary(SGridData<double> &velocity) const;
        /// 扩散算子使用的齐次 Neumann 条件；语义化别名便于调用端阅读。
        void ApplyDiffusionNeumann(GridData<double> &field) const;

        /// 判断该面是否位于与其速度分量同方向的域边界上。
        bool IsNormalBoundaryFace(int axis, Vector2i const &face) const;
        /// 判断单元/面是否属于去除 boundary band 后的物理计算域。
        bool IsInsideCell(Vector2i const &cell) const;
        bool IsInsideFace(int axis, Vector2i const &face) const;
        /// 将连续世界坐标限制到物理域的闭包，供半拉格朗日回溯使用。
        Vector2d ClampPositionToDomain(Vector2d const &position) const;
        DomainBoundaryType GetType() const { return m_Type; }

    private:
        /// 将任意边界带坐标截取到最近的物理内部单元。
        Vector2i ClampToInteriorCell(Vector2i const &cell) const;

        /// 网格生命周期由 Simulation 保证长于本对象。
        StaggeredGrid const &m_Grid;
        DomainBoundaryType m_Type;
    };
}
