#ifndef LINSCALE_TABLE_H_
#define LINSCALE_TABLE_H_

#include "matrix_mkl.h"
#include <map>
#include <vector>

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
std::map<T, std::vector<int>>
buildRangeIndex(const Matrix<T, majorOrder>& mat, int numJoinAttrs)
{
    std::map<T, std::vector<int>> mRangeIndex;

    for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
    {
        T tVal = mat(rowIdx, 0);
        if (mRangeIndex.contains(tVal))
        {
            mRangeIndex[tVal].push_back(rowIdx);
        }
        else
        {
            mRangeIndex[tVal] = {rowIdx};
        }
    }
    return mRangeIndex;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> sortTable(const Matrix<T, majorOrder>& mat, int numSortAttrs)
{
    Matrix<T, majorOrder> matOut{mat.getNumRows(), mat.getNumCols()};

    auto mOut = buildRangeIndex(mat, numSortAttrs);
    int outRowIdx = 0;
    for (const auto& [val, vRowInds]: mOut)
    {
        for (auto& rowIdx: vRowInds)
        {
            for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
            {
                matOut(outRowIdx, colIdx) = mat(rowIdx, colIdx);
            }
            outRowIdx++;
        }
    }
    return std::move(matOut);
}


template <typename T, MajorOrder majorOrder>
int computeJoinSize(Matrix<T, majorOrder>& mat1, Matrix<T, majorOrder>& mat2,
    const std::map<T, std::vector<int>>& rangeIdx1,
    const std::map<T, std::vector<int>>& rangeIdx2)
{
    int joinSize = 0;
    for (const auto& [val, vRowIdxs]: rangeIdx1)
    {
        int curJoinSize = 0;
        if (rangeIdx2.contains(val))
        {
            curJoinSize = rangeIdx2.at(val).size() * vRowIdxs.size();
        }
        joinSize += curJoinSize;
    }

    return joinSize;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder>
computeJoin(Matrix<T, majorOrder>& mat1, Matrix<T, majorOrder>& mat2, int numJoinAttrs)
{
    auto rangeIdx1 = buildRangeIndex(mat1, numJoinAttrs);
    auto rangeIdx2 = buildRangeIndex(mat2, numJoinAttrs);
    auto joinSize = computeJoinSize(mat1, mat2, rangeIdx1, rangeIdx2);

    Matrix<T, majorOrder> matOut{joinSize, mat1.getNumCols() + mat2.getNumCols() - numJoinAttrs};
    int outRowIdx = 0;
    for (const auto& [val, rowIdxs1]: rangeIdx1)
    {
        for (auto& rowIdx1: rowIdxs1)
        {
            if (not rangeIdx2.contains(val))
            {
                continue;
            }

            const auto& rowIdxs2 = rangeIdx2[val];
            for (auto& rowIdx2: rowIdxs2)
            {
                for (int colIdx = 0; colIdx < mat1.getNumCols(); colIdx++)
                {
                    matOut(outRowIdx, colIdx) = mat1(rowIdx1, colIdx);
                }

                for (int colIdx = mat1.getNumCols() + numJoinAttrs; colIdx < mat1.getNumCols() + mat2.getNumCols(); colIdx++)
                {
                    matOut(outRowIdx, colIdx - numJoinAttrs) = mat2(rowIdx2, colIdx - mat1.getNumCols());
                }
                outRowIdx++;
            }
        }
    }

    return matOut;
}

#endif