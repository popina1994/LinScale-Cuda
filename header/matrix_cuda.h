#ifndef _LINSCALE_MATRIX_CUDA_H_
#define _LINSCALE_MATRIX_CUDA_H_

#include "matrix.h"
#include "cuda_util.h"
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

    MatrixCuda<T, majorOrder> copyMatrix(int startRowIdx, int endRowIdx, int startColIdx, int endColIdx);

    MatrixCuda<T, majorOrder> computeInverse(void)
    {
        //
    }

    MatrixCuda<T, majorOrder> multiply(const MatrixCuda<T, majorOrder>& mat2)
    {
        auto& mat1 = *this;
        // dgmem
    }

    int computeQRDecomposition(MatrixCuda<T, MajorOrder::COL_MAJOR>& matR,
        MatrixCuda<T, MajorOrder::COL_MAJOR>& matQ, bool computeQ = false);
};


template <typename T>
using MatrixCudaCol = MatrixCuda<T, MajorOrder::COL_MAJOR>;
template <typename T>
using MatrixCudaRow = MatrixCuda<T, MajorOrder::ROW_MAJOR>;

// Outside of the class (not a member function)
template <typename T, MajorOrder majorOrder>
__global__ void copyMatrixCuda(const T* d_ASrc, T* d_Bdst, int numRowsSrc, int numColsSrc, int startRowIdx, int endRowIdx, int startColIdx, int endColIdx) {
    int colIdxDst  = threadIdx.x;
    int rowIdxDst = blockIdx.x;
    int colIdxSrc = threadIdx.x + startColIdx;
    int rowIdxSrc = blockIdx.x + startRowIdx;
    int numRowsDst = endRowIdx - startRowIdx + 1;
    int numColsDst = endColIdx - startColIdx + 1;

    if (rowIdxSrc < numRowsSrc and rowIdxSrc < numColsSrc)
    {
        int posIdxSrc;
        int posIdxDst;
        if (majorOrder == MajorOrder::COL_MAJOR)
        {
            posIdxSrc = IDX_C(rowIdxSrc, colIdxSrc, numRowsSrc, numColsSrc);
            posIdxDst = IDX_C(rowIdxDst, colIdxDst, numRowsDst, numColsDst);
        }
        else
        {
            posIdxSrc = IDX_R(rowIdxSrc, colIdxSrc, numRowsSrc, numColsSrc);
            posIdxDst = IDX_R(rowIdxDst, colIdxDst, numRowsDst, numColsDst);
        }
        d_Bdst[posIdxDst] = d_ASrc[posIdxSrc];
    }
}

template <typename T>
__global__ void setZerosUpperTriangularCol(T* d_A, int numRows, int numColsSrc) {
    int colIdx = threadIdx.x;
    int rowIdx = blockIdx.x;
    if (rowIdx > colIdx and rowIdx < numRows and colIdx < numColsSrc)
    {
        int posIdx = IDX_C(rowIdx, colIdx, numRows, numColsSrc);
        d_A[posIdx] = 0;
    }
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder> MatrixCuda<T, majorOrder>::
copyMatrix(int startRowIdx, int endRowIdx, int startColIdx, int endColIdx)
{
    MatrixCuda<T, majorOrder> matOut{endRowIdx - startRowIdx + 1, endColIdx - startColIdx + 1};
    copyMatrixCuda<T, majorOrder><<<matOut.getNumRows(), matOut.getNumCols()>>>
        (getDataC(), matOut.getData(), getNumRows(), getNumCols(), startRowIdx, endRowIdx, startColIdx, endColIdx);
    return matOut;
}

template <typename T, MajorOrder majorOrder>
int MatrixCuda<T, majorOrder>::computeQRDecomposition(MatrixCudaCol<T>& matR,
        MatrixCuda<T, MajorOrder::COL_MAJOR>& matQ, bool computeQ)
{
    if (majorOrder == MajorOrder::COL_MAJOR)
    {
        auto& matA = *this;
        cusolverDnHandle_t cuSolverHandle;
        CUSOLVER_CALL(cusolverDnCreate(&cuSolverHandle));
        int workspace_size = 0;
        if constexpr (std::is_same<T, float>::value)
        {
            CUSOLVER_CALL(cusolverDnSgeqrf_bufferSize(cuSolverHandle, getNumRows(), getNumCols(), getData(), getNumRows(), &workspace_size));
        }
        else
        {
            CUSOLVER_CALL(cusolverDnDgeqrf_bufferSize(cuSolverHandle, getNumRows(), getNumCols(), getData(), getNumRows(), &workspace_size));
        }
        // Allocate workspace
        T *d_work, *d_tau;
        CUDA_CALL(cudaMalloc((void**)&d_work, workspace_size * sizeof(T)));

        // Allocate device status variable
        int *devInfo;
        CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)));
        CUDA_CALL(cudaMalloc((void**)&d_tau, std::min(getNumRows(), getNumCols()) * sizeof(T)));
        if constexpr (std::is_same<T, float>::value)
        {
            CUSOLVER_CALL(cusolverDnSgeqrf(cuSolverHandle, getNumRows(), getNumCols(), getData(), getNumRows(), d_tau, d_work, workspace_size, devInfo));
        }
        else
        {
            CUSOLVER_CALL(cusolverDnDgeqrf (cuSolverHandle, getNumRows(), getNumCols(), getData(), getNumRows(), d_tau, d_work, workspace_size, devInfo));
        }
        matR = copyMatrix(0, getNumCols() - 1, 0, getNumCols() - 1);
        setZerosUpperTriangularCol<<<matR.getNumRows(), matR.getNumCols()>>>(matR.getData(), matR.getNumRows(), matR.getNumCols());

        CUDA_CALL(cudaFree(d_tau));
        CUDA_CALL(cudaFree(d_work));
        CUDA_CALL(cudaFree(devInfo));
    }
    return 0;
}


template<typename T>
MatrixCudaCol<T> changeLayoutFromRowToColumn(const MatrixCudaRow<T>& matA)
{
    MatrixCudaCol<T> matCudaCol(matA.getNumRows(), matA.getNumCols());
    int numRows = matA.getNumRows();
    int numCols = matA.getNumCols();

    cublasHandle_t handle;
    cublasCreate(&handle);

    const T alpha = 1.0f; // Scalar for matrix A (no scaling)
    const T beta = 0.0f;  // Scalar for matrix B (no B matrix, so no scaling)

    if constexpr (std::is_same<T, float>::value)
    {
        cublasSgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
        numRows, numCols,                     // Matrix dimensions
        &alpha,                   // Scalar for A
        matA.getDataC(), numCols,                 // Input matrix A and its leading dimension
        &beta,                    // Scalar for B (not used)
        nullptr, numCols,               // No B matrix (set to nullptr)
        matCudaCol.getData(), numRows);                  // Output matrix C and its leading dimension
    }
    else
    {
        cublasDgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
        numRows, numCols,                     // Matrix dimensions
        &alpha,                   // Scalar for A
        matA.getDataC(), numCols,                   // Input matrix A and its leading dimension
        &beta,                    // Scalar for B (not used)
        nullptr, numCols,               // No B matrix (set to nullptr)
        matCudaCol.getData(), numRows);                  // Output matrix C and its leading dimension
    }
    return matCudaCol;
}
template <typename T>
void printDeviceVector(const thrust::device_vector<T>& dVector)
{
    thrust::host_vector<T> hVector(dVector);
    std::cout << "Device vector " << dVector.size() << std::endl;
    for (auto& elem: hVector)
    {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

#endif