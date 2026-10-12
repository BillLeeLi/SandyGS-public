#include "physics/SandConstitutive.h"

namespace Pivot
{
    namespace
    {
        bool IsFinite(Matrix2d const &matrix)
        {
            return matrix.array().isFinite().all();
        }

        /// 二分搜索最大的 alpha，使沿 strain(alpha) 得到的状态仍在屈服面内。
        template <typename StrainFunction, typename ResidualFunction>
        double FindElasticScale(
            StrainFunction const &strainAt,
            ResidualFunction const &residualAt)
        {
            double lower = 0.0;
            double upper = 1.0;
            for (int iteration = 0; iteration < 80; ++iteration)
            {
                double const middle = 0.5 * (lower + upper);
                if (residualAt(strainAt(middle)) <= 0.0)
                {
                    lower = middle;
                }
                else
                {
                    upper = middle;
                }
            }
            return lower;
        }
    }

    SandConstitutive::SandConstitutive(
        PhysicalParameters const &parameters,
        double determinantEpsilon)
        : parameters_(parameters),
          determinantEpsilon_(determinantEpsilon)
    {
        if (!(determinantEpsilon_ > 0.0) ||
            !std::isfinite(determinantEpsilon_))
        {
            throw std::invalid_argument(
                "constitutive determinant epsilon must be finite and positive");
        }
    }

    Matrix2d SandConstitutive::ComputeCauchyStress(
        Matrix2d const &elasticF,
        double saturation) const
    {
        // 饱和度只移动屈服面，不改变首版使用的弹性模量。
        (void)saturation;
        double const determinant = elasticF.determinant();
        if (!IsFinite(elasticF) || determinant <= determinantEpsilon_)
        {
            throw std::invalid_argument(
                "Cauchy stress requires a finite orientation-preserving F");
        }

        JacobiSVD<Matrix2d> const svd(
            elasticF, ComputeFullU | ComputeFullV);
        Vector2d const singularValues =
            svd.singularValues().cwiseMax(determinantEpsilon_);
        Vector2d const logarithmicStrain =
            singularValues.array().log().matrix();

        // psi=mu||epsilon||^2+lambda/2 tr(epsilon)^2。
        // tau_i=d psi/d epsilon_i，sigma=tau/J。
        Vector2d const principalCauchy =
            (2.0 * parameters_.lameMu * logarithmicStrain +
             parameters_.lameLambda *
                 logarithmicStrain.sum() * Vector2d::Ones()) /
            determinant;
        Matrix2d const stress =
            svd.matrixU() * principalCauchy.asDiagonal() *
            svd.matrixU().transpose();
        return 0.5 * (stress + stress.transpose());
    }

    double SandConstitutive::YieldResidual(
        Matrix2d const &stress,
        double saturation) const
    {
        if (!IsFinite(stress))
        {
            throw std::invalid_argument(
                "yield residual requires a finite stress");
        }
        double const trace = stress.trace();
        Matrix2d const deviator =
            stress - 0.5 * trace * Matrix2d::Identity();
        return parameters_.druckerPragerFriction * trace +
            deviator.norm() -
            EvaluateCohesion(saturation);
    }

    Matrix2d SandConstitutive::Project(
        Matrix2d const &trialF,
        double saturation) const
    {
        bool ignored = false;
        return Project(trialF, saturation, ignored);
    }

    Matrix2d SandConstitutive::Project(
        Matrix2d const &trialF,
        double saturation,
        bool &changed) const
    {
        if (!IsFinite(trialF) ||
            trialF.determinant() <= determinantEpsilon_)
        {
            throw std::invalid_argument(
                "plastic projection requires a finite orientation-preserving F");
        }

        JacobiSVD<Matrix2d> const svd(
            trialF, ComputeFullU | ComputeFullV);
        Matrix2d const U = svd.matrixU();
        Matrix2d const V = svd.matrixV();
        Vector2d const trialLogStrain =
            svd.singularValues()
                .cwiseMax(determinantEpsilon_)
                .array()
                .log()
                .matrix();

        auto const residualFromStrain = [&](Vector2d const &strain)
        {
            // M7 代数快速路径：candidate=U exp(strain) V^T 的 Hencky 主应力
            // 可直接由 strain 和 U 得到，无需在二分搜索的每次迭代中重建矩阵
            // 后再做一次 SVD。公式与 ComputeCauchyStress() 完全相同。
            double const determinant =
                std::exp(strain.sum());
            Vector2d const principalCauchy =
                (2.0 * parameters_.lameMu * strain +
                 parameters_.lameLambda *
                     strain.sum() * Vector2d::Ones()) /
                determinant;
            Matrix2d const stress =
                U * principalCauchy.asDiagonal() * U.transpose();
            return YieldResidual(
                0.5 * (stress + stress.transpose()),
                saturation);
        };
        if (residualFromStrain(trialLogStrain) <= 0.0)
        {
            changed = false;
            return trialF;
        }
        changed = true;

        double const mean = 0.5 * trialLogStrain.sum();
        Vector2d const deviator =
            trialLogStrain - mean * Vector2d::Ones();
        Vector2d returnedStrain;

        // 首先保持体积对数应变，只沿偏应变方向径向返回。这一选择与论文
        // 引用的 Drucker--Prager radial return 一致，同时保留可压缩响应。
        Vector2d const hydrostatic = mean * Vector2d::Ones();
        if (residualFromStrain(hydrostatic) <= 0.0)
        {
            auto const strainAt = [&](double scale)
            {
                return hydrostatic + scale * deviator;
            };
            double const scale = FindElasticScale(
                strainAt, residualFromStrain);
            returnedStrain = strainAt(scale);
        }
        else
        {
            // 纯静水拉伸也可能越过屈服面；此时偏应变已归零，再把体积应变
            // 朝无应变状态收回。零应变的残差为 -cohesion，必定位于屈服面内。
            auto const strainAt = [&](double scale)
            {
                return scale * hydrostatic;
            };
            double const scale = FindElasticScale(
                strainAt, residualFromStrain);
            returnedStrain = strainAt(scale);
        }

        Matrix2d const projected =
            ReconstructFromLogStrain(U, V, returnedStrain);
        if (!IsFinite(projected) ||
            projected.determinant() <= determinantEpsilon_)
        {
            throw std::runtime_error(
                "plastic projection produced an invalid deformation gradient");
        }
        return projected;
    }

    void SandConstitutive::PolarDecompose(
        Matrix2d const &F,
        double &rotationAngle,
        Matrix2d &stretch)
    {
        if (!IsFinite(F) || F.determinant() <= 0.0)
        {
            throw std::invalid_argument(
                "polar decomposition requires det(F)>0");
        }
        JacobiSVD<Matrix2d> const svd(F, ComputeFullU | ComputeFullV);
        Matrix2d const rotation =
            svd.matrixU() * svd.matrixV().transpose();
        stretch =
            svd.matrixV() * svd.singularValues().asDiagonal() *
            svd.matrixV().transpose();
        stretch = 0.5 * (stretch + stretch.transpose());
        rotationAngle = std::atan2(rotation(1, 0), rotation(0, 0));
    }

    double SandConstitutive::EvaluateCohesion(double saturation) const
    {
        double const phi = std::clamp(saturation, 0.0, 1.0);
        CohesionParameters const &c = parameters_.cohesion;
        if (phi < c.phi1)
        {
            return c.c0 + phi * (c.c1 - c.c0) / c.phi1;
        }
        if (phi < c.phi2)
        {
            return c.c1 +
                (phi - c.phi1) * (c.c2 - c.c1) /
                    (c.phi2 - c.phi1);
        }
        if (phi <= c.phi3)
        {
            return c.c2 * (c.phi3 - phi) /
                (c.phi3 - c.phi2);
        }
        return 0.0;
    }

    Matrix2d SandConstitutive::ReconstructFromLogStrain(
        Matrix2d const &U,
        Matrix2d const &V,
        Vector2d const &logStrain)
    {
        return U * logStrain.array().exp().matrix().asDiagonal() *
            V.transpose();
    }
}
