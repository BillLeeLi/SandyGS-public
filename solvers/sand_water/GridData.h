#pragma once

#include "Grid.h"

namespace Pivot
{
    /***
     * 网格上的标量场
     */
    template <typename Type>
    class GridData
    {
    public:
        explicit GridData(Grid const &grid, Type const &value = Zero<Type>()) : 
        m_Grid{grid},
        m_Data(grid.GetNumVertices(), value) {
        }

        GridData &operator=(GridData const &rhs) // 相同尺寸网格之间的赋值
        {
            if (m_Grid == rhs.m_Grid)
            {
                m_Data = rhs.m_Data;
            }
            else
            {
                spdlog::critical("Failed to assign between GridData with different grids");
                std::exit(EXIT_FAILURE);
            }
            return *this;
        }

        Grid const &GetGrid() const { return m_Grid; }
        auto &GetData() { return m_Data; }
        auto const &GetData() const { return m_Data; }

        Type &operator[](Vector2i const &coord) { return m_Data[m_Grid.IndexOf(coord)]; }
        Type const &operator[](Vector2i const &coord) const { return m_Data[m_Grid.IndexOf(coord)]; }
        Type const &At(Vector2i const &coord) const { return m_Data[m_Grid.IndexOf(m_Grid.Clamp(coord))]; }  // 相比operator[]，At函数会将坐标限制在网格范围内
        Type &operator[](int index) { return m_Data[index]; }
        Type const &operator[](int index) const { return m_Data[index]; }

        Type GetMaxAbsValue() const
            requires(std::is_arithmetic_v<Type>)
        {
            if (m_Data.empty())
            {
                return 0;
            }
            else
            {
                auto minmax = std::minmax_element(m_Data.begin(), m_Data.end());
                return std::max(std::abs(*minmax.first), std::abs(*minmax.second));
            }
        }

        void SetConstant(Type const &value) { std::fill(m_Data.begin(), m_Data.end(), value); }
        void SetZero() { SetConstant(Zero<Type>()); }

        void Save(std::ostream &out) const { IO::Write(out, m_Data); }
        void Load(std::istream &in) { IO::Read(in, m_Data); }

    private:
        Grid const &m_Grid;
        std::vector<Type> m_Data; // 网格中数据按照行优先顺序存储
    };
}
