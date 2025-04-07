#ifndef _LINSCALE_MATRIX_CUDA_H_
#define _LINSCALE_MATRIX_CUDA_H_

#include "matrix.h"
#include <string>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

template <typename T, MajorOrder majorOrder>
class MatrixCuda
{
    int numRows = 0;
    int numCols = 0;
    T* pArr = nullptr;
    thrust::device_vector<T> dVector;

    MatrixCuda(int _numRows, int _numCols, const thrust::device_vector<T>& mdVector): numRows(_numRows), numCols(_numCols), dVector(mdVector)
    {
        // std::cout << "CREATE" << pArr << std::endl;
    }
public:
    MatrixCuda(const MatrixCuda& matIn) = delete;
    MatrixCuda& operator=(const MatrixCuda& matIn) = delete;
    MatrixCuda(int _numRows, int _numCols): numRows(_numRows), numCols(_numCols), dVector(int64_t(numRows) * int64_t(numCols))
    {
        // std::cout << "CREATE" << pArr << std::endl;
    }

    MatrixCuda(const Matrix<T, majorOrder>& matHost): numRows(matHost.getNumRows()), numCols(matHost.getNumCols()),
        dVector(matHost.getDataC(), matHost.getDataC() + matHost.getNumElements())
    {
        // std::cout << "CREATE" << pArr << std::endl;
    }


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

    int getSize(void) const
    {
        return numRows * numCols * sizeof(T);
    }

    MatrixCuda<T, majorOrder> getColumn(int colIdx) const
    {
        thrust::device_vector<T> outVect;

        if constexpr (majorOrder == MajorOrder::COL_MAJOR)
        {
            outVect = std::move(thrust::device_vector<T>(getDataC(), getDataC() + getNumRows()));
        }
        else
        {
            int numColsCur = numCols;
            auto isNthElement = [numColsCur] __device__ (int idx) { return (idx %  numColsCur == 0); };
            auto onlyEvenIter = thrust::make_counting_iterator(0);
            outVect = thrust::device_vector<T>(numRows);
            thrust::copy_if(
                dVector.begin() + colIdx,
                dVector.end(),
                onlyEvenIter,
                outVect.begin(),
                isNthElement);
        }
        MatrixCuda<T, majorOrder> matOut(getNumRows(), 1, outVect);
        return matOut;
    }

    Matrix<T, majorOrder> getHostCopy(void) const
    {
        thrust::host_vector<T> hostVector = dVector;;
        Matrix<T, majorOrder> matOut(thrust::raw_pointer_cast(hostVector.data()), getNumRows(), getNumCols());
        return matOut;
    }

    friend std::ostream& operator<<(std::ostream& out, MatrixCuda<T, majorOrder>& matCuda)
    {
        Matrix<T, majorOrder> matHost = matCuda.getHostCopy();

        out << matHost;

        return out;
    }
};


template <typename T>
using MatrixCudaCol = MatrixCuda<T, MajorOrder::COL_MAJOR>;
template <typename T>
using MatrixCudaRow = MatrixCuda<T, MajorOrder::ROW_MAJOR>;

#endif