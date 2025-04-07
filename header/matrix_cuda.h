#ifndef _LINSCALE_MATRIX_CUDA_H_
#define _LINSCALE_MATRIX_CUDA_H_

#include "matrix.h"
#include <string>
#include <thrust/device_vector.h>

template <typename T, MajorOrder majorOrder>
struct MatrixCuda
{
    int numRows = 0;
    int numCols = 0;
    T* pArr = nullptr;
    thrust::device_vector<T> dVector;
    MatrixCuda(int _numRows, int _numCols): numRows(_numRows), numCols(_numCols), dVector(int64_t(numRows) * int64_t(numCols))
    {
        // std::cout << "CREATE" << pArr << std::endl;
    }

    MatrixCuda(const Matrix<T, majorOrder>& matHost): numRows(matHost.getNumRows()), numCols(matHost.getNumCols()),
        dVector(matHost.getDataC(), matHost.getDataC() + matHost.getNumElements())
    {
        // std::cout << "CREATE" << pArr << std::endl;
    }

    MatrixCuda(const MatrixCuda& matIn) = delete;
    MatrixCuda& operator=(const MatrixCuda& matIn) = delete;
    MatrixCuda(MatrixCuda&& matIn)
    {
        dVector = std::move(matIn.dVector);
        // std::cout << "MOVE " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
    }

    MatrixCuda& operator=(MatrixCuda&& matIn)
    {
        dVector = std::move(matIn.dVector);
        // std::cout << "ASSIGN " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        return *this;
    }

    ~MatrixCuda()
    {}

    T& operator()(int rowIdx, int colIdx)
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return dVector[posId];
    }

    const T& operator()(int rowIdx, int colIdx) const
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return dVector[posId];
    }
    T* getData()
    {
        return thrust::raw_pointer_cast(dVector.data());
    }

    const T* getDataC() const
    {
        return thrust::raw_pointer_cast(dVector.data());
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

#endif