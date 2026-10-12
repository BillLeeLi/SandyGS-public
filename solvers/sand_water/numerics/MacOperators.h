#pragma once

#include "BiLerp.h"
#include "SGridData.h"
#include "StaggeredGrid.h"

namespace Pivot
{
    /// MAC 网格上单元中心量与面中心量之间的常用离散算子。
    ///
    /// 物理模块通过这些方法访问相邻单元，避免在多个 Solver 中重复手写容易
    /// 颠倒左右顺序的 i±1 索引。调用者应只传入具有两个有效邻接单元的面。
    class MacOperators
    {
    public:
        /// 将面左右两个单元中心标量做算术平均。
        static double AverageToFace(
            GridData<double> const &field,
            int axis,
            Vector2i const &face)
        {
            Vector2i const left = StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const right = StaggeredGrid::AdjCellOfFace(axis, face, 1);
            return 0.5 * (field[left] + field[right]);
        }

        /// 将面左右两个单元中心矩阵逐分量平均。
        static Matrix2d AverageToFace(
            GridData<Matrix2d> const &field,
            int axis,
            Vector2i const &face)
        {
            Vector2i const left = StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const right = StaggeredGrid::AdjCellOfFace(axis, face, 1);
            return 0.5 * (field[left] + field[right]);
        }

        /// 单元中心标量沿 axis 的面中心一阶差分。
        static double GradientAtFace(
            GridData<double> const &field,
            int axis,
            Vector2i const &face)
        {
            Vector2i const left = StaggeredGrid::AdjCellOfFace(axis, face, 0);
            Vector2i const right = StaggeredGrid::AdjCellOfFace(axis, face, 1);
            return (field[right] - field[left]) * field.GetGrid().GetInvSpacing();
        }

        /// 在一个 MAC 面的位置重建完整二维速度；法向分量精确取该面值。
        static Vector2d VelocityAtFace(
            SGridData<double> const &velocity,
            int axis,
            Vector2i const &face)
        {
            Vector2d const position = velocity[axis].GetGrid().PositionOf(face);
            Vector2d result = BiLerp::Interpolate(velocity, position);
            result[axis] = velocity[axis][face];
            return result;
        }

        /// 在单元中心计算二维速度梯度，供后续沙体形变更新复用。
        static Matrix2d VelocityGradientAtCell(
            SGridData<double> const &velocity,
            Vector2i const &cell)
        {
            double const invDx = velocity.GetGrids()[0].GetInvSpacing();
            Matrix2d gradient;
            gradient(0, 0) =
                (velocity[0][cell + Vector2i::UnitX()] - velocity[0][cell]) * invDx;
            gradient(1, 1) =
                (velocity[1][cell + Vector2i::UnitY()] - velocity[1][cell]) * invDx;

            // 交叉项先将相应速度分量平均到上下/左右单元面，再做中心差分。
            gradient(0, 1) = 0.25 * invDx * (
                velocity[0][cell + Vector2i::UnitY()] +
                velocity[0][cell + Vector2i::UnitX() + Vector2i::UnitY()] -
                velocity[0][cell - Vector2i::UnitY()] -
                velocity[0][cell + Vector2i::UnitX() - Vector2i::UnitY()]);
            gradient(1, 0) = 0.25 * invDx * (
                velocity[1][cell + Vector2i::UnitX()] +
                velocity[1][cell + Vector2i::UnitX() + Vector2i::UnitY()] -
                velocity[1][cell - Vector2i::UnitX()] -
                velocity[1][cell - Vector2i::UnitX() + Vector2i::UnitY()]);
            return gradient;
        }
    };
}
