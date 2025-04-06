#ifndef _LIN_SCALE_MATRIX_H_
#define _LIN_SCALE_MATRIX_H_

#include "types.h"
template <typename T, MajorOrder majorOrder>
struct Matrix
{
    int numRows = 0;
    int numCols = 0;
    T* pArr = nullptr;
    Matrix(int _numRows, int _numCols): numRows(_numRows), numCols(_numCols)
    {
        pArr = new T[int64_t(numRows) * int64_t(numCols)];
        // std::cout << "CREATE" << pArr << std::endl;
    }

    Matrix(const Matrix& matIn) = delete;
    Matrix& operator=(const Matrix& matIn) = delete;
    Matrix(Matrix&& matIn)
    {
        if (pArr != nullptr)
        {
            delete [] pArr;
        }
        pArr = matIn.pArr;
        // std::cout << "MOVE " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        matIn.pArr = nullptr;
    }

    Matrix& operator=(Matrix&& matIn)
    {
        if (pArr != nullptr)
        {
            delete [] pArr;
        }
        pArr = matIn.pArr;
        // std::cout << "ASSIGN " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        matIn.pArr = nullptr;
        return *this;
    }

    ~Matrix()
    {
        if (pArr != nullptr)
        {
            // std::cout << "DELETE" << pArr  << std::endl;
            delete [] pArr;
            pArr = nullptr;
        }
    }

    T& operator()(int rowIdx, int colIdx)
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return pArr[posId];
    }

    const T& operator()(int rowIdx, int colIdx) const
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return pArr[posId];
    }
    T*& getData()
    {
        return pArr;
    }

    const T* getDataC() const
    {
        return pArr;
    }

    int getNumRows(void) const {
        return numRows;

    }
    int getNumCols(void) const
    {
        return numCols;
    }

    int getNumElements(void) const
    {
        return numRows * numCols;
    }
};

using MatrixDCol = Matrix<double, MajorOrder::COL_MAJOR>;
using MatrixDRow = Matrix<double, MajorOrder::ROW_MAJOR>;

template <typename T, MajorOrder order>
void printMatrix(const T* pArr, int numRows, int numCols, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?" << fileName << std::endl;
    }
    for (int rowIdx = 0; rowIdx < std::min(numRows, numRowsCut); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numCols; colIdx++)
        {
            if (colIdx > 0)
            {
                outFile << ",";
            }
            if (upperTriangular and (rowIdx > colIdx))
            {
                outFile << "0";
            }
            else
            {
                if constexpr (order == MajorOrder::ROW_MAJOR)
                {
                    outFile << pArr[IDX_R(rowIdx, colIdx, numRows, numCols)];
                }
                else
                {
                    outFile << pArr[IDX_C(rowIdx, colIdx, numRows, numCols)];
                }
            }

        }
        outFile << std::endl;
    }
}

template <typename T, MajorOrder order>
void printMatrix(const Matrix<T, order>& matA, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?  " << fileName << std::endl;
    }
    for (int rowIdx = 0; rowIdx < std::min(matA.getNumRows(), numRowsCut); rowIdx++)
    {
        for (int colIdx = 0; colIdx < matA.getNumCols(); colIdx++)
        {
            if (colIdx > 0)
            {
                outFile << ",";
            }
            if (upperTriangular and (rowIdx > colIdx))
            {
                outFile << "0";
            }
            else
            {
                outFile << matA(rowIdx, colIdx);
            }

        }
        outFile << std::endl;
    }
}
#endif