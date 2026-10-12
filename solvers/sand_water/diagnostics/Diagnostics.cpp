#include "diagnostics/Diagnostics.h"
#include "physics/SandConstitutive.h"

namespace Pivot
{
    namespace
    {
        bool IsFinite(Matrix2d const &matrix)
        {
            return matrix.array().isFinite().all();
        }

        std::runtime_error FieldError(
            std::string_view stage,
            std::string_view field,
            Vector2i const &coord,
            std::string_view reason)
        {
            return std::runtime_error(fmt::format(
                "{}: field '{}' at ({}, {}) {}",
                stage, field, coord.x(), coord.y(), reason));
        }

        Vector2d CellVelocity(
            SGridData<double> const &velocity,
            Vector2i const &cell)
        {
            // 将包围单元的同向两个 MAC 面速度做算术平均，得到单元中心速度。
            return {
                0.5 * (velocity[0][cell] + velocity[0][cell + Vector2i::UnitX()]),
                0.5 * (velocity[1][cell] + velocity[1][cell + Vector2i::UnitY()])};
        }

        double MaxAbsVelocity(SGridData<double> const &velocity)
        {
            return velocity.GetMaxAbsComponent();
        }
    }

    ConservationReport Diagnostics::Measure(
        SimulationState const &state,
        SimulationConfig const &config,
        StaggeredGrid const &grid) const
    {
        ConservationReport report;
        double const area = grid.GetSpacing() * grid.GetSpacing();
        report.minHeight = std::numeric_limits<double>::infinity();
        report.minDetF = std::numeric_limits<double>::infinity();
        bool hasSand = false;
        SandConstitutive const constitutive(
            config.physics,
            config.time.determinantEpsilon);

        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            // ghost/boundary cells 只服务于差分模板，不能计入物理守恒量。
            if (!grid.IsInsideCell(cell))
            {
                return;
            }

            double const sandHeight = state.sand.height[cell];
            double const pureHeight = state.pureWater.height[cell];
            double const mixedHeight = state.mixedWater.height[cell];

            report.sandVolume += sandHeight * area;
            report.pureWaterVolume += pureHeight * area;
            report.mixedWaterVolume += mixedHeight * area;
            report.minHeight =
                std::min({report.minHeight, sandHeight, pureHeight, mixedHeight});

            report.horizontalMomentum +=
                // 2.5D 高度场中单元质量为 density * columnHeight * cellArea。
                config.physics.sandDensity * sandHeight * area *
                    CellVelocity(state.sand.velocity, cell) +
                config.physics.waterDensity * pureHeight * area *
                    CellVelocity(state.pureWater.velocity, cell) +
                config.physics.waterDensity * mixedHeight * area *
                    CellVelocity(state.mixedWater.velocity, cell);

            if (sandHeight > config.time.heightEpsilon)
            {
                hasSand = true;
                report.minDetF =
                    std::min(report.minDetF, state.sand.deformationGradient[cell].determinant());
                double const water = pureHeight + mixedHeight;
                double const saturation =
                    water / (water + sandHeight);
                Matrix2d const stress =
                    constitutive.ComputeCauchyStress(
                        state.sand.deformationGradient[cell],
                        saturation);
                // 只记录屈服面外的正残差；合法投影状态和无沙场均报告 0。
                report.maxYieldResidual = std::max(
                    report.maxYieldResidual,
                    std::max(
                        constitutive.YieldResidual(
                            stress, saturation),
                        0.0));
            }
        });

        report.totalWaterVolume = report.pureWaterVolume + report.mixedWaterVolume;
        if (!std::isfinite(report.minHeight))
        {
            report.minHeight = 0.0;
        }
        if (!hasSand)
        {
            report.minDetF = 1.0;
        }

        report.maxVelocity = std::max({
            MaxAbsVelocity(state.sand.velocity),
            MaxAbsVelocity(state.pureWater.velocity),
            MaxAbsVelocity(state.mixedWater.velocity)});
        report.timeStep = state.lastDt;
        report.cfl = report.maxVelocity * state.lastDt * grid.GetInvSpacing();
        report.sandCfl =
            state.sand.velocity.GetMaxAbsComponent() *
            state.lastDt /
            static_cast<double>(config.time.sandSubsteps) *
            grid.GetInvSpacing();
        return report;
    }

    void Diagnostics::Validate(
        SimulationState const &state,
        SimulationConfig const &config,
        StaggeredGrid const &grid,
        BoundaryConditions const &boundary,
        std::string_view stage) const
    {
        auto const validateLayer = [&](LayerState const &layer, std::string_view name)
        {
            ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
            {
                double const height = layer.height[cell];
                if (!std::isfinite(height))
                {
                    throw FieldError(stage, name, cell, "is not finite");
                }
                if (grid.IsInsideCell(cell) && height < -config.time.heightEpsilon)
                {
                    throw FieldError(stage, name, cell, "is negative");
                }
            });

            ForEach(grid.GetFaceGrids(), [&](int axis, Vector2i const &face)
            {
                double const value = layer.velocity[axis][face];
                if (!std::isfinite(value))
                {
                    throw FieldError(
                        stage, fmt::format("{}Velocity{}", name, axis), face, "is not finite");
                }
                if (boundary.IsNormalBoundaryFace(axis, face) &&
                    std::abs(value) > config.time.heightEpsilon)
                {
                    // 任一材料穿过封闭边界都会破坏质量守恒，因此立即终止。
                    throw FieldError(
                        stage, fmt::format("{}Velocity{}", name, axis), face,
                        "violates the closed normal boundary");
                }
            });
        };

        ForEach(grid.GetCellGrid(), [&](Vector2i const &cell)
        {
            if (!std::isfinite(state.terrain[cell]))
            {
                throw FieldError(stage, "terrain", cell, "is not finite");
            }

            if (grid.IsInsideCell(cell) &&
                state.mixedWater.height[cell] >
                    state.sand.height[cell] + config.time.heightEpsilon)
            {
                // 这是当前高度场分层表示的核心几何不变量。
                throw FieldError(stage, "mixedWaterHeight", cell, "exceeds sand height");
            }

            Matrix2d const &F = state.sand.deformationGradient[cell];
            Matrix2d const &P = state.sand.stretch[cell];
            double const theta = state.sand.rotationAngle[cell];
            if (!IsFinite(F) || !IsFinite(P) || !std::isfinite(theta))
            {
                throw FieldError(stage, "sandConstitutiveState", cell, "is not finite");
            }
            if (grid.IsInsideCell(cell) &&
                state.sand.height[cell] > config.time.heightEpsilon &&
                F.determinant() <= config.time.determinantEpsilon)
            {
                // 只检查活跃沙单元；空单元的本构状态会由 ResetConstitutiveState 清理。
                throw FieldError(stage, "deformationGradient", cell, "has a non-positive determinant");
            }
        });

        validateLayer(state.sand, "sand");
        validateLayer(state.pureWater, "pureWater");
        validateLayer(state.mixedWater, "mixedWater");
    }
}
