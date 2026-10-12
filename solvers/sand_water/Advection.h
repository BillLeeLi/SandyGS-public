#pragma once

#include "BiLerp.h"
#include "numerics/BoundaryConditions.h"

namespace Pivot
{
    /// 守恒高度平流后用于测试和运行时诊断的质量摘要。
    struct AdvectionStats
    {
        double volumeBefore{0.0};
        double volumeAfter{0.0};
        double minValue{0.0};

        double VolumeChange() const { return volumeAfter - volumeBefore; }
    };

    /// 单元中心守恒量的一阶迎风有限体积平流。
    ///
    /// 每个面通量只读取 source 和 n 时刻速度，destination 单独写入。高度版本
    /// 额外使用 donor 流出限制器：若一个时间步计划流出的水超过单元现有水量，
    /// 就等比例缩小该单元所有流出面通量，因而同时保持非负性和面通量守恒。
    class AdvectionFV
    {
    public:
        /// 使用法向速度选择迎风侧；正速度表示从 left 流向 right。
        template <typename Type>
        static Type ComputeFaceFlux(
            Type const &left,
            Type const &right,
            double normalVelocity)
        {
            return normalVelocity >= 0.0
                ? left * normalVelocity
                : right * normalVelocity;
        }

        /// 平流一般单元中心守恒量，例如后续阶段使用的 h*theta 和 h*P。
        ///
        /// 该通用版本不对 Type 定义“非负”，因此不使用高度流出限制器；调用者
        /// 必须保证相应守恒量与承载它的高度使用一致通量。M1 的水高应调用
        /// SolveHeight()，不能用这个接口绕过非负保护。
        template <typename Type>
        static void SolveConserved(
            GridData<Type> const &source,
            SGridData<double> const &velocity,
            double dt,
            BoundaryConditions const &boundary,
            GridData<Type> &destination)
        {
            if (&source == &destination)
            {
                throw std::invalid_argument(
                    "finite-volume advection requires distinct source and destination");
            }
            if (dt < 0.0)
            {
                throw std::invalid_argument("advection time step must be non-negative");
            }

            double const invDx = source.GetGrid().GetInvSpacing();
            // 不使用 SGridData<Type>，因为该容器的默认构造值面向算术标量；
            // h*P 等矩阵守恒量只需要两个独立的面网格场。
            GridData<Type> fluxX(velocity.GetGrids()[0]);
            GridData<Type> fluxY(velocity.GetGrids()[1]);
            auto fluxAt = [&](int axis) -> GridData<Type> &
            {
                return axis == 0 ? fluxX : fluxY;
            };
            ParallelForEach(velocity.GetGrids(), [&](int axis, Vector2i const &face)
            {
                Vector2i const left = StaggeredGrid::AdjCellOfFace(axis, face, 0);
                Vector2i const right = StaggeredGrid::AdjCellOfFace(axis, face, 1);
                if (boundary.IsNormalBoundaryFace(axis, face) ||
                    !boundary.IsInsideCell(left) ||
                    !boundary.IsInsideCell(right))  // 只对内部域单元的面计算通量
                {
                    fluxAt(axis)[face] = Zero<Type>();
                    return;
                }
                fluxAt(axis)[face] = ComputeFaceFlux(
                    source[left], source[right], velocity[axis][face]);
            });

            destination = source;
            ParallelForEach(source.GetGrid(), [&](Vector2i const &cell)
            {
                if (!boundary.IsInsideCell(cell))
                {
                    return;
                }
                Type divergence = Zero<Type>();
                for (int axis = 0; axis < 2; ++axis)
                {
                    divergence +=
                        fluxAt(axis)[cell + Vector2i::Unit(axis)] -
                        fluxAt(axis)[cell];
                }
                destination[cell] = source[cell] - dt * invDx * divergence;
            });
        }

        /// 使用 carrier 的 donor 流出限制器平流附着在该材料上的守恒量。
        ///
        /// M4 对 h*theta、h*P 调用此方法，并以同一 h 另行调用 SolveHeight。
        /// 两者会得到逐面完全相同的 donor scale，因此即使 CFL 保护被触发，
        /// 本构量也不会与承载它的沙质量分离。
        template <typename Type>
        static void SolveConservedWithCarrier(
            GridData<Type> const &source,
            GridData<double> const &carrier,
            SGridData<double> const &velocity,
            double dt,
            BoundaryConditions const &boundary,
            GridData<Type> &destination)
        {
            if (&source == &destination)
            {
                throw std::invalid_argument(
                    "finite-volume advection requires distinct source and destination");
            }
            if (dt < 0.0)
            {
                throw std::invalid_argument(
                    "advection time step must be non-negative");
            }

            double const invDx = carrier.GetGrid().GetInvSpacing();
            GridData<double> outgoingScale(carrier.GetGrid(), 1.0);
            ParallelForEach(carrier.GetGrid(), [&](Vector2i const &cell)
            {
                if (!boundary.IsInsideCell(cell))
                {
                    outgoingScale[cell] = 0.0;
                    return;
                }
                double outgoingRate = 0.0;
                for (int ordinal = 0;
                     ordinal < StaggeredGrid::GetNumFacesPerCell();
                     ++ordinal)
                {
                    auto const [axis, face] =
                        StaggeredGrid::FaceOfCell(cell, ordinal);
                    if (boundary.IsNormalBoundaryFace(axis, face))
                    {
                        continue;
                    }
                    int const side =
                        StaggeredGrid::FaceSideOfCell(ordinal);
                    double const outwardVelocity =
                        velocity[axis][face] * side;
                    if (outwardVelocity > 0.0)
                    {
                        outgoingRate +=
                            outwardVelocity *
                            std::max(carrier[cell], 0.0);
                    }
                }
                double const requested =
                    dt * invDx * outgoingRate;
                double const available =
                    std::max(carrier[cell], 0.0);
                if (requested > available && requested > 0.0)
                {
                    outgoingScale[cell] = available / requested;
                }
            });

            GridData<Type> fluxX(velocity.GetGrids()[0]);
            GridData<Type> fluxY(velocity.GetGrids()[1]);
            auto fluxAt = [&](int axis) -> GridData<Type> &
            {
                return axis == 0 ? fluxX : fluxY;
            };
            ParallelForEach(velocity.GetGrids(),
                [&](int axis, Vector2i const &face)
            {
                Vector2i const left =
                    StaggeredGrid::AdjCellOfFace(axis, face, 0);
                Vector2i const right =
                    StaggeredGrid::AdjCellOfFace(axis, face, 1);
                if (boundary.IsNormalBoundaryFace(axis, face) ||
                    !boundary.IsInsideCell(left) ||
                    !boundary.IsInsideCell(right))
                {
                    fluxAt(axis)[face] = Zero<Type>();
                    return;
                }
                double const normalVelocity =
                    velocity[axis][face];
                Vector2i const donor =
                    normalVelocity >= 0.0 ? left : right;
                fluxAt(axis)[face] =
                    source[donor] * normalVelocity *
                    outgoingScale[donor];
            });

            destination = source;
            ParallelForEach(source.GetGrid(), [&](Vector2i const &cell)
            {
                if (!boundary.IsInsideCell(cell))
                {
                    return;
                }
                Type divergence = Zero<Type>();
                for (int axis = 0; axis < 2; ++axis)
                {
                    divergence +=
                        fluxAt(axis)[cell + Vector2i::Unit(axis)] -
                        fluxAt(axis)[cell];
                }
                destination[cell] =
                    source[cell] - dt * invDx * divergence;
            });
        }

        /// 平流非负高度场并返回物理内部域的体积变化统计。
        static AdvectionStats SolveHeight(
            GridData<double> const &source,
            SGridData<double> const &velocity,
            double dt,
            BoundaryConditions const &boundary,
            GridData<double> &destination)
        {
            if (&source == &destination)
            {
                throw std::invalid_argument(
                    "finite-volume advection requires distinct source and destination");
            }
            if (dt < 0.0)
            {
                throw std::invalid_argument("advection time step must be non-negative");
            }

            Grid const &cellGrid = source.GetGrid();
            double const invDx = cellGrid.GetInvSpacing();
            double const area = cellGrid.GetSpacing() * cellGrid.GetSpacing();

            // outgoingScale(i,j) 属于 donor 单元，并同时作用于它的所有流出面。
            // 这比更新后直接 clamp 负高度更重要：后者会凭空增加质量。
            GridData<double> outgoingScale(cellGrid, 1.0);
            ParallelForEach(cellGrid, [&](Vector2i const &cell)
            {
                if (!boundary.IsInsideCell(cell))
                {
                    outgoingScale[cell] = 0.0;
                    return;
                }

                double outgoingRate = 0.0;
                for (int ordinal = 0; ordinal < StaggeredGrid::GetNumFacesPerCell(); ++ordinal)
                {
                    auto const [axis, face] = StaggeredGrid::FaceOfCell(cell, ordinal);
                    if (boundary.IsNormalBoundaryFace(axis, face))
                    {
                        continue;
                    }

                    int const side = StaggeredGrid::FaceSideOfCell(ordinal);
                    double const outwardVelocity = velocity[axis][face] * side;
                    if (outwardVelocity > 0.0)
                    {
                        outgoingRate += outwardVelocity * std::max(source[cell], 0.0);
                    }
                }

                double const requestedOutflow = dt * invDx * outgoingRate;
                double const available = std::max(source[cell], 0.0);
                if (requestedOutflow > available && requestedOutflow > 0.0)
                {
                    outgoingScale[cell] = available / requestedOutflow;
                }
            });

            // flux[axis][face] 的正号始终表示沿 axis 正方向输运高度。
            SGridData<double> flux(velocity.GetGrids());
            ParallelForEach(velocity.GetGrids(), [&](int axis, Vector2i const &face)
            {
                Vector2i const left = StaggeredGrid::AdjCellOfFace(axis, face, 0);
                Vector2i const right = StaggeredGrid::AdjCellOfFace(axis, face, 1);
                if (boundary.IsNormalBoundaryFace(axis, face) ||
                    !boundary.IsInsideCell(left) ||
                    !boundary.IsInsideCell(right))
                {
                    flux[axis][face] = 0.0;
                    return;
                }

                double const normalVelocity = velocity[axis][face];
                if (normalVelocity >= 0.0)
                {
                    flux[axis][face] =
                        source[left] * normalVelocity * outgoingScale[left];
                }
                else
                {
                    flux[axis][face] =
                        source[right] * normalVelocity * outgoingScale[right];
                }
            });

            destination = source;
            ParallelForEach(cellGrid, [&](Vector2i const &cell)
            {
                if (!boundary.IsInsideCell(cell))
                {
                    return;
                }

                double divergence = 0.0;
                for (int axis = 0; axis < 2; ++axis)
                {
                    Vector2i const upperFace = cell + Vector2i::Unit(axis);
                    divergence += flux[axis][upperFace] - flux[axis][cell];
                }
                double const updated = source[cell] - dt * invDx * divergence;

                // 限制器理论上保证 updated >= 0；这里只消除舍入级负零，并对真正
                // 的负值立即报错，避免用截断掩盖通量方向或索引错误。
                if (updated < -64.0 * std::numeric_limits<double>::epsilon() *
                                  std::max(1.0, std::abs(source[cell])))
                {
                    throw std::runtime_error(fmt::format(
                        "finite-volume advection produced negative height {} at ({}, {})",
                        updated, cell.x(), cell.y()));
                }
                destination[cell] = std::max(updated, 0.0);
            });

            AdvectionStats stats;
            stats.minValue = std::numeric_limits<double>::infinity();
            ForEach(cellGrid, [&](Vector2i const &cell)
            {
                if (!boundary.IsInsideCell(cell))
                {
                    return;
                }
                stats.volumeBefore += source[cell] * area;
                stats.volumeAfter += destination[cell] * area;
                stats.minValue = std::min(stats.minValue, destination[cell]);
            });
            if (!std::isfinite(stats.minValue))
            {
                stats.minValue = 0.0;
            }
            return stats;
        }
    };

    /// MAC 面速度的二阶中点回溯半拉格朗日平流。
    ///
    /// 该方法不严格守恒动量，但对速度场稳定且与论文采用的方法一致。调用者应
    /// 在进入这里之前把有效水面速度向干区外推，并在完成后重新施加边界条件。
    class AdvectionSL
    {
    public:
        /// 从 startPosition 沿 flow 反向追踪 dt；中点速度提供二阶轨迹精度。
        static Vector2d TraceMidpoint(
            Vector2d const &startPosition,
            SGridData<double> const &flow,
            double dt,
            BoundaryConditions const &boundary)
        {
            Vector2d const start = boundary.ClampPositionToDomain(startPosition);
            Vector2d const velocity0 = BiLerp::Interpolate(flow, start);
            Vector2d const midpoint = boundary.ClampPositionToDomain(
                start - 0.5 * dt * velocity0);
            Vector2d const velocityMid = BiLerp::Interpolate(flow, midpoint);
            return boundary.ClampPositionToDomain(start - dt * velocityMid);
        }

        /// 用 flow 回溯并采样 source 的两个 MAC 分量。
        static void Solve(
            SGridData<double> const &source,
            SGridData<double> const &flow,
            double dt,
            BoundaryConditions const &boundary,
            SGridData<double> &destination)
        {
            if (&source == &destination)
            {
                throw std::invalid_argument(
                    "semi-Lagrangian advection requires distinct source and destination");
            }
            if (dt < 0.0)
            {
                throw std::invalid_argument("advection time step must be non-negative");
            }

            destination = source;
            ParallelForEach(source.GetGrids(), [&](int axis, Vector2i const &face)
            {
                if (!boundary.IsInsideFace(axis, face))
                {
                    return;
                }
                Vector2d const position = source[axis].GetGrid().PositionOf(face);
                Vector2d const departure =
                    TraceMidpoint(position, flow, dt, boundary);
                destination[axis][face] =
                    BiLerp::Interpolate(source[axis], departure);
            });
        }
    };
}
