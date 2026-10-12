#include "numerics/BoundaryConditions.h"

namespace Pivot
{
    BoundaryConditions::BoundaryConditions(
        StaggeredGrid const &grid,
        DomainBoundaryType type)
        : m_Grid(grid),
          m_Type(type)
    {
    }

    Vector2i BoundaryConditions::ClampToInteriorCell(Vector2i const &cell) const
    {
        int const width = m_Grid.GetBoundaryWidth();
        Vector2i const lo = Vector2i::Constant(width);
        Vector2i const hi = m_Grid.GetResolution() - Vector2i::Constant(width + 1);
        return cell.cwiseMax(lo).cwiseMin(hi);
    }

    void BoundaryConditions::UpdateMasks(
        SimulationState const &state,
        double heightEpsilon,
        SimulationScratch &scratch) const
    {
        ParallelForEach(m_Grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            // IsInsideCell 排除 boundary band；即使 ghost cell 从内部复制到正高度，
            // 也不能被数值算子误认为实际材料。
            bool const inside = m_Grid.IsInsideCell(cell);
            scratch.sandActive[cell] =
                static_cast<std::uint8_t>(inside && state.sand.height[cell] > heightEpsilon);
            scratch.pureWaterActive[cell] =
                static_cast<std::uint8_t>(inside && state.pureWater.height[cell] > heightEpsilon);
            scratch.mixedWaterActive[cell] =
                static_cast<std::uint8_t>(inside && state.mixedWater.height[cell] > heightEpsilon);
        });

        BuildFaceMask(scratch.sandActive, scratch.faceValid);
    }

    void BoundaryConditions::BuildFaceMask(
        GridData<std::uint8_t> const &cellMask,
        SGridData<std::uint8_t> &faceMask) const
    {
        ParallelForEach(m_Grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            // MAC 面夹在两个单元之间。只要一侧有材料，该面速度就可能参与
            // 通量或外推；两侧都为空时可以跳过昂贵计算。
            bool valid = false;
            for (int side = 0; side < 2; ++side)
            {
                Vector2i const cell = StaggeredGrid::AdjCellOfFace(axis, face, side);
                if (m_Grid.GetCellGrid().IsValid(cell) && cellMask[cell])
                {
                    valid = true;
                }
            }
            faceMask[axis][face] = static_cast<std::uint8_t>(valid);
        });
    }

    void BoundaryConditions::BuildWaterTerrainOpenMask(
        SimulationState const &state,
        double heightEpsilon,
        SGridData<std::uint8_t> &faceOpen) const
    {
        ParallelForEach(m_Grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            Vector2i const negative =
                StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const positive =
                StaggeredGrid::AdjCellOfFace(axis, face, 1);
            // 矩形域边界由既有边界条件处理。这里只判断拥有两个物理邻居的
            // 内部面，避免把 ghost-cell 高度误当成场景中的地形障碍。
            if (!IsInsideCell(negative) || !IsInsideCell(positive))
            {
                faceOpen[axis][face] = 1;
                return;
            }

            double const negativeWater =
                state.pureWater.height[negative] +
                state.mixedWater.height[negative];
            double const positiveWater =
                state.pureWater.height[positive] +
                state.mixedWater.height[positive];
            bool const negativeWet = negativeWater > heightEpsilon;
            bool const positiveWet = positiveWater > heightEpsilon;
            double const negativeSurface =
                state.terrain[negative] + negativeWater;
            double const positiveSurface =
                state.terrain[positive] + positiveWater;

            // 重要：干湿判断使用总水深，而不是 pure/mixed 各自高度。否则
            // 两层相互转换的界面会被错误识别为实体墙。
            bool const blockedByNegativeDryHighTerrain =
                !negativeWet && positiveWet &&
                state.terrain[negative] > positiveSurface;
            bool const blockedByPositiveDryHighTerrain =
                !positiveWet && negativeWet &&
                state.terrain[positive] > negativeSurface;
            faceOpen[axis][face] = static_cast<std::uint8_t>(
                !blockedByNegativeDryHighTerrain &&
                !blockedByPositiveDryHighTerrain);
        });
    }

    void BoundaryConditions::BuildSandTerrainOpenMask(
        SimulationState const &state,
        double heightEpsilon,
        double contactTolerance,
        SGridData<std::uint8_t> &faceOpen) const
    {
        ParallelForEach(m_Grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            Vector2i const negative =
                StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const positive =
                StaggeredGrid::AdjCellOfFace(axis, face, 1);
            if (!IsInsideCell(negative) || !IsInsideCell(positive))
            {
                faceOpen[axis][face] = 1;
                return;
            }

            double const negativeSand = state.sand.height[negative];
            double const positiveSand = state.sand.height[positive];
            bool const negativeActive = negativeSand > heightEpsilon;
            bool const positiveActive = positiveSand > heightEpsilon;
            double const negativeSandSurface =
                state.terrain[negative] + negativeSand;
            double const positiveSandSurface =
                state.terrain[positive] + positiveSand;

            // 沙的水压载荷 rho_w/rho_s*hpure 是动力学势的一部分，但不会
            // 提高沙柱真实几何顶部，因此越障只比较 H 与 H+hs。这里给沙面
            // 增加一个仅用于接触分类的小容差，使摄影测量地形的微小毛刺不
            // 会成为人工墙；真正超过该容差的高地仍保持封闭。
            bool const blockedByNegativeDryHighTerrain =
                !negativeActive && positiveActive &&
                state.terrain[negative] >
                    positiveSandSurface + contactTolerance;
            bool const blockedByPositiveDryHighTerrain =
                !positiveActive && negativeActive &&
                state.terrain[positive] >
                    negativeSandSurface + contactTolerance;
            faceOpen[axis][face] = static_cast<std::uint8_t>(
                !blockedByNegativeDryHighTerrain &&
                !blockedByPositiveDryHighTerrain);
        });
    }

    void BoundaryConditions::IntersectFaceMasks(
        SGridData<std::uint8_t> const &materialFaceValid,
        SGridData<std::uint8_t> const &terrainFaceOpen,
        SGridData<std::uint8_t> &destination) const
    {
        ParallelForEach(m_Grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            bool const valid =
                materialFaceValid[axis][face] != 0 &&
                terrainFaceOpen[axis][face] != 0;
            destination[axis][face] = static_cast<std::uint8_t>(valid);
        });
    }

    void BoundaryConditions::ApplyTerrainContactBoundary(
        SGridData<double> &velocity,
        SGridData<std::uint8_t> const &terrainFaceOpen) const
    {
        ParallelForEach(m_Grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            if (!terrainFaceOpen[axis][face])
            {
                velocity[axis][face] = 0.0;
            }
        });
    }

    void BoundaryConditions::ApplyHeightBoundary(GridData<double> &field) const
    {
        if (m_Grid.GetBoundaryWidth() == 1)
        {
            // 一层边界的所有采样点都严格位于内部，永远不会读取本循环正在写的
            // ghost cell，因此可以原地更新并省去一次完整 GridData 复制。
            ParallelForEach(field.GetGrid(), [&](Vector2i const &cell)
            {
                if (!m_Grid.IsInsideCell(cell))
                {
                    field[cell] = field[ClampToInteriorCell(cell)];
                }
            });
            return;
        }

        // 必须从不可变快照采样。若直接原地逐层复制，多层边界带的结果会依赖
        // 并行执行顺序。
        GridData<double> const source = field;
        ParallelForEach(field.GetGrid(), [&](Vector2i const &cell)
        {
            if (!m_Grid.IsInsideCell(cell))
            {
                field[cell] = source[ClampToInteriorCell(cell)];
            }
        });
    }

    bool BoundaryConditions::IsNormalBoundaryFace(int axis, Vector2i const &face) const
    {
        int const width = m_Grid.GetBoundaryWidth();
        int const resolution = m_Grid.GetResolution()[axis];
        return face[axis] <= width || face[axis] >= resolution - width;
    }

    bool BoundaryConditions::IsInsideCell(Vector2i const &cell) const
    {
        return m_Grid.IsInsideCell(cell);
    }

    bool BoundaryConditions::IsInsideFace(int axis, Vector2i const &face) const
    {
        return m_Grid.IsInsideFace(axis, face);
    }

    Vector2d BoundaryConditions::ClampPositionToDomain(Vector2d const &position) const
    {
        Vector2d const lower = m_Grid.GetDomainOrigin();
        Vector2d const upper = lower + m_Grid.GetDomainLengths();
        return position.cwiseMax(lower).cwiseMin(upper);
    }

    void BoundaryConditions::ApplyVelocityBoundary(SGridData<double> &velocity) const
    {
        if (m_Type == DomainBoundaryType::ClosedNoSlip)
        {
            // no-slip 只写零而不读取旧场，任何边界宽度都不需要快照。
            ParallelForEach(m_Grid.GetFaceGrids(),
                [&](int axis, Vector2i const &face)
            {
                if (m_Grid.IsBoundaryFace(axis, face))
                {
                    velocity[axis][face] = 0.0;
                }
            });
            return;
        }

        bool const canUpdateInPlace =
            m_Grid.GetBoundaryWidth() == 1;
        // 自由滑移需要从未修改的切向速度采样，因此同样保留输入快照。
        // 一层边界时，切向 ghost 始终读取未被修改的内部切向面，可安全原地读。
        std::optional<SGridData<double>> source;
        if (!canUpdateInPlace)
        {
            source.emplace(velocity);
        }
        int const width = m_Grid.GetBoundaryWidth();
        Vector2i const resolution = m_Grid.GetResolution();

        ParallelForEach(m_Grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
        {
            if (IsNormalBoundaryFace(axis, face))
            {
                // 封闭域的不可穿透条件：位于左右边界的 u 和上下边界的 v 为零。
                velocity[axis][face] = 0.0;
                return;
            }

            int const tangent = axis ^ 1;
            if (face[tangent] < width || face[tangent] >= resolution[tangent] - width)
            {
                // 对切向分量复制最近内部面，等价于 dv_tangent/dn = 0。
                Vector2i sample = face;
                sample[tangent] =
                    std::clamp(sample[tangent], width, resolution[tangent] - width - 1);
                velocity[axis][face] = canUpdateInPlace
                    ? velocity[axis][sample]
                    : (*source)[axis][sample];
            }
        });
    }

    void BoundaryConditions::ApplyDiffusionNeumann(GridData<double> &field) const
    {
        ApplyHeightBoundary(field);
    }
}
