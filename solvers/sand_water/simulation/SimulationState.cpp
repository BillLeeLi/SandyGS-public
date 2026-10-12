#include "simulation/SimulationState.h"

namespace Pivot
{
    LayerState::LayerState(StaggeredGrid const &grid)
        : height(grid.GetCellGrid()),
          velocity(grid.GetFaceGrids())
    {
    }

    void LayerState::SetZero()
    {
        height.SetZero();
        velocity.SetZero();
    }

    SandState::SandState(StaggeredGrid const &grid)
        : LayerState(grid),
          deformationGradient(grid.GetCellGrid(), Matrix2d::Identity()),
          rotationAngle(grid.GetCellGrid(), 0.0),
          stretch(grid.GetCellGrid(), Matrix2d::Identity())
    {
    }

    void SandState::SetZero()
    {
        LayerState::SetZero();
        deformationGradient.SetConstant(Matrix2d::Identity());
        rotationAngle.SetZero();
        stretch.SetConstant(Matrix2d::Identity());
    }

    void SandState::ResetConstitutiveState(GridData<std::uint8_t> const &active)
    {
        if (!(deformationGradient.GetGrid() == active.GetGrid()))
        {
            throw std::invalid_argument("constitutive active mask uses a different grid");
        }

        ParallelForEach(deformationGradient.GetGrid(), [&](Vector2i const &cell)
        {
            // 材料离开单元后不应遗留应变历史，否则该单元再次被沙占据时
            // 会凭空继承旧应力。活跃单元的本构历史则必须原样保留。
            if (!active[cell])
            {
                deformationGradient[cell] = Matrix2d::Identity();
                rotationAngle[cell] = 0.0;
                stretch[cell] = Matrix2d::Identity();
            }
        });
    }

    SimulationState::SimulationState(StaggeredGrid const &grid)
        : terrain(grid.GetCellGrid()),
          sand(grid),
          pureWater(grid),
          mixedWater(grid)
    {
    }

    void SimulationState::SetZero()
    {
        terrain.SetZero();
        sand.SetZero();
        pureWater.SetZero();
        mixedWater.SetZero();
        time = 0.0;
        lastDt = 0.0;
        stepIndex = 0;
    }

    SimulationScratch::SimulationScratch(StaggeredGrid const &grid)
        // 所有缓存一次性按最终网格分配。后续 Step() 中只覆盖内容，不改变尺寸，
        // 从而让各数值阶段没有隐藏的堆内存分配。
        : scalar0(grid.GetCellGrid()),
          scalar1(grid.GetCellGrid()),
          totalWater(grid.GetCellGrid()),
          mixedHeightDelta(grid.GetCellGrid()),
          oldSandHeight(grid.GetCellGrid()),
          saturation(grid.GetCellGrid()),
          rawConstitutiveSaturation(grid.GetCellGrid()),
          hydraulicPotential(grid.GetCellGrid()),
          sandPotential(grid.GetCellGrid()),
          matrix0(grid.GetCellGrid()),
          matrix1(grid.GetCellGrid()),
          stress(grid.GetCellGrid()),
          weightedStress(grid.GetCellGrid()),
          internalYieldResidual(grid.GetCellGrid()),
          postAdvectionYieldResidualBefore(grid.GetCellGrid()),
          postAdvectionYieldResidualAfter(grid.GetCellGrid()),
          postAdvectionProjectionDelta(grid.GetCellGrid()),
          velocity0(grid.GetFaceGrids()),
          waterTransferImpulse(grid.GetFaceGrids()),
          sandWaterImpulse(grid.GetFaceGrids()),
          internalAcceleration(grid.GetFaceGrids()),
          internalNormalAcceleration(grid.GetFaceGrids()),
          internalTangentialAcceleration(grid.GetFaceGrids()),
          internalFaceHeight(grid.GetFaceGrids()),
          internalEffectiveHeight(grid.GetFaceGrids()),
          sandActive(grid.GetCellGrid()),
          pureWaterActive(grid.GetCellGrid()),
          mixedWaterActive(grid.GetCellGrid()),
          sandFaceValid(grid.GetFaceGrids()),
          pureWaterFaceValid(grid.GetFaceGrids()),
          mixedWaterFaceValid(grid.GetFaceGrids()),
          waterTerrainFaceOpen(grid.GetFaceGrids()),
          sandTerrainFaceOpen(grid.GetFaceGrids()),
          faceValid(grid.GetFaceGrids()),
          constitutiveVisited(
              static_cast<std::size_t>(
                  grid.GetCellGrid().GetNumVertices()),
              0)
    {
        constitutiveFrontier.reserve(
            static_cast<std::size_t>(
                grid.GetCellGrid().GetNumVertices()));
    }

    void SimulationScratch::SetZero()
    {
        scalar0.SetZero();
        scalar1.SetZero();
        totalWater.SetZero();
        mixedHeightDelta.SetZero();
        oldSandHeight.SetZero();
        saturation.SetZero();
        rawConstitutiveSaturation.SetZero();
        hydraulicPotential.SetZero();
        sandPotential.SetZero();

        matrix0.SetZero();
        matrix1.SetZero();
        stress.SetZero();
        weightedStress.SetZero();
        internalYieldResidual.SetZero();
        postAdvectionYieldResidualBefore.SetZero();
        postAdvectionYieldResidualAfter.SetZero();
        postAdvectionProjectionDelta.SetZero();

        velocity0.SetZero();
        waterTransferImpulse.SetZero();
        sandWaterImpulse.SetZero();
        internalAcceleration.SetZero();
        internalNormalAcceleration.SetZero();
        internalTangentialAcceleration.SetZero();
        internalFaceHeight.SetZero();
        internalEffectiveHeight.SetZero();

        sandActive.SetZero();
        pureWaterActive.SetZero();
        mixedWaterActive.SetZero();
        sandFaceValid.SetZero();
        pureWaterFaceValid.SetZero();
        mixedWaterFaceValid.SetZero();
        waterTerrainFaceOpen.SetZero();
        sandTerrainFaceOpen.SetZero();
        faceValid.SetZero();
        std::fill(
            constitutiveVisited.begin(),
            constitutiveVisited.end(),
            std::uint8_t{0});
        constitutiveFrontier.clear();
    }
}
