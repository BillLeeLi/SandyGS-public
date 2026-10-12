#pragma once

#include "simulation/SimulationConfig.h"

namespace Pivot
{
    /// 论文中的二维 Hencky 弹性与含水 Modified Drucker--Prager 塑性模型。
    ///
    /// 该类型不保存网格状态，只保存材料参数，因此既可由 SandSolver 复用，
    /// 也可以脱离完整模拟对单个 F 做确定性的本构单元测试。
    class SandConstitutive
    {
    public:
        explicit SandConstitutive(
            PhysicalParameters const &parameters,
            double determinantEpsilon);

        /// 将 trial F 返回到屈服面内；弹性状态保持不变，且结果始终 det(F)>0。
        Matrix2d Project(
            Matrix2d const &trialF,
            double saturation) const;

        /// 与 Project() 相同，并通过 changed 告知调用者返回映射是否修改了 F。
        ///
        /// SandSolver 用该标志跳过弹性状态上冗余的第二次极分解；false 时返回值
        /// 与 trialF 是同一个矩阵值，而不是近似相等。
        Matrix2d Project(
            Matrix2d const &trialF,
            double saturation,
            bool &changed) const;

        /// 由弹性变形梯度计算二维 Cauchy 应力。
        Matrix2d ComputeCauchyStress(
            Matrix2d const &elasticF,
            double saturation) const;

        /// f=cf tr(sigma)+||dev(sigma)||-cc(phi)；f<=0 表示未屈服。
        double YieldResidual(
            Matrix2d const &stress,
            double saturation) const;

        /// 极分解 F=R(theta)P；P 为对称正定右拉伸张量。
        static void PolarDecompose(
            Matrix2d const &F,
            double &rotationAngle,
            Matrix2d &stretch);

    private:
        /// 严格实现论文图 3 的四段含水黏聚力曲线。
        double EvaluateCohesion(double saturation) const;

        /// 用固定的奇异向量和给定主对数应变重建 F。
        static Matrix2d ReconstructFromLogStrain(
            Matrix2d const &U,
            Matrix2d const &V,
            Vector2d const &logStrain);

        PhysicalParameters parameters_;
        /// 奇异值和行列式的共同正数下界。
        double determinantEpsilon_;
    };
}
