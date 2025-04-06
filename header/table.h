#ifndef LINSCALE_TABLE_H_
#define LINSCALE_TABLE_H_

#include "types.h"
#include "matrix_mkl.h"

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder>
generateRandomJoinTable(int numRows, int numCols, int numJoinAttrs, int numJoinVals, int seed)
{
    std::mt19937 gen(seed); // Fixed seed
    std::uniform_real_distribution<T> distReal(0.0, 1.0);

    std::vector<int> values(numJoinVals);
    std::iota(values.begin(), values.end(), 1);
    std::vector<double> weights (numJoinVals, 1.0 / numJoinVals);
    std::discrete_distribution<> disInt(weights.begin(), weights.end());

    auto matA = std::move(Matrix<T, majorOrder>{numRows, numCols});

    // col_major
    if constexpr (MajorOrder::COL_MAJOR == majorOrder)
    {
        // TODO Add generation for join attributes
        for (int colIdx = 0; colIdx < numJoinAttrs; colIdx++)
        {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
            {
                int joinAttr = disInt(gen);
                matA(rowIdx, colIdx) =  values[joinAttr];
            }
        }
        for (int colIdx = numJoinAttrs; colIdx < numCols; colIdx++)
        {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
            {
                matA(rowIdx, colIdx) = distReal(gen);
            }
        }
    }
    else
    {
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        {
            for (int colIdx = 0; colIdx < numJoinAttrs; colIdx++)
            {
                int joinAttr = disInt(gen);
                matA(rowIdx, colIdx) =  values[joinAttr];
            }
            for (int colIdx = numJoinAttrs; colIdx < numCols; colIdx++)
            {
                matA(rowIdx, colIdx) = distReal(gen);
            }
        }
    }
    return std::move(matA);
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder>
computeJoin(Matrix<T, order>& mat1, Matrix<T, order>& mat2, int numJoinAttrs)
{
    std::mt19937 gen(seed); // Fixed seed
    std::uniform_real_distribution<T> distReal(0.0, 1.0);

    std::vector<int> values(numJoinVals);
    std::iota(values.begin(), values.end(), 1);
    std::vector<double> weights (numJoinVals, 1.0 / numJoinVals);
    std::discrete_distribution<> disInt(weights.begin(), weights.end());

    auto matA = std::move(Matrix<T, majorOrder>{numRows, numCols});

    // col_major
    if constexpr (MajorOrder::COL_MAJOR == majorOrder)
    {
        // TODO Add generation for join attributes
        for (int colIdx = 0; colIdx < numJoinAttrs; colIdx++)
        {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
            {
                int joinAttr = disInt(gen);
                matA(rowIdx, colIdx) =  values[joinAttr];
            }
        }
        for (int colIdx = numJoinAttrs; colIdx < numCols; colIdx++)
        {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
            {
                matA(rowIdx, colIdx) = distReal(gen);
            }
        }
    }
    else
    {
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        {
            for (int colIdx = 0; colIdx < numJoinAttrs; colIdx++)
            {
                int joinAttr = disInt(gen);
                matA(rowIdx, colIdx) =  values[joinAttr];
            }
            for (int colIdx = numJoinAttrs; colIdx < numCols; colIdx++)
            {
                matA(rowIdx, colIdx) = distReal(gen);
            }
        }
    }
    return std::move(matA);
}

#endif