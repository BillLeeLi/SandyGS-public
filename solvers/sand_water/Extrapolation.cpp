#include "Extrapolation.h"

namespace Pivot
{
    void Extrapolation::ExtrapolateRotationStretchOneRing3x3(
        GridData<double> &rotationAngle,
        GridData<Matrix2d> &stretch,
        GridData<std::uint8_t> const &tracked)
    {
        if (!(rotationAngle.GetGrid() == stretch.GetGrid()) ||
            !(rotationAngle.GetGrid() == tracked.GetGrid()))
        {
            throw std::invalid_argument(
                "rotation/stretch extrapolation fields use different grids");
        }

        GridData<double> const sourceAngle = rotationAngle;
        GridData<Matrix2d> const sourceStretch = stretch;
        ParallelForEach(rotationAngle.GetGrid(), [&](Vector2i const &cell)
        {
            if (tracked[cell])
            {
                return;
            }

            int count = 0;
            double reference = 0.0;
            double angleSum = 0.0;
            Matrix2d stretchSum = Matrix2d::Zero();
            for (int offsetX = -1; offsetX <= 1; ++offsetX)
            {
                for (int offsetY = -1; offsetY <= 1; ++offsetY)
                {
                    Vector2i const neighbor =
                        cell + Vector2i(offsetX, offsetY);
                    if (!tracked.GetGrid().IsValid(neighbor) ||
                        !tracked[neighbor])
                    {
                        continue;
                    }
                    if (count == 0)
                    {
                        reference = sourceAngle[neighbor];
                    }
                    // remainder 给出 [-pi,pi] 内与 reference 最近的等价角。
                    angleSum += reference + std::remainder(
                        sourceAngle[neighbor] - reference,
                        2.0 * std::numbers::pi);
                    stretchSum += sourceStretch[neighbor];
                    ++count;
                }
            }

            if (count > 0)
            {
                rotationAngle[cell] = angleSum / count;
                stretch[cell] =
                    0.5 * (stretchSum + stretchSum.transpose()) / count;
            }
            else
            {
                rotationAngle[cell] = 0.0;
                stretch[cell] = Matrix2d::Identity();
            }
        });
    }

    void Extrapolation::Solve(GridData<double> &grData, double clearVal, int maxSteps, GridData<std::uint8_t> &valid)
    {
        auto valid1 = valid;
        for (int iter = 0; iter < maxSteps; iter++)
        {
            ParallelForEach(grData.GetGrid(), [&](Vector2i const &coord)
                            {
				if (!valid[coord]) {
                    // 对于无效点，计算其邻居的平均值作为外推值
                    // 如果没有有效邻居，使用clearVal作为外推值
					int    cnt = 0;
					double sum = 0;
					for (int i = 0; i < Grid::GetNumNeighbors(); i++) {
						Vector2i const nbCoord = Grid::NeighborOf(coord, i);
						if (grData.GetGrid().IsValid(nbCoord) && valid[nbCoord]) {
							sum += grData[nbCoord];
							cnt++;
						}
					}
					if (cnt > 0) {
						grData[coord] = sum / cnt;
						valid1[coord] = true;
					} else {
						grData[coord] = clearVal;
					}
				} else {
					valid1[coord] = true;
				} });
            valid.GetData().swap(valid1.GetData());
        }
    }
}
