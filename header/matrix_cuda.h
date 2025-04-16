#ifndef _LINSCALE_MATRIX_CUDA_H_
#define _LINSCALE_MATRIX_CUDA_H_

#include "matrix.h"
#include "cuda_util.h"
#include <string>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

template <typename T, MajorOrder majorOrder>
class MatrixCuda
{
    int numRows = 0;
    int numCols = 0;
    T* pArr = nullptr;
    thrust::device_vector<T> dVector;

    MatrixCuda(int _numRows, int _numCols, const thrust::device_vector<T>& mdVector): numRows(_numRows), numCols(_numCols), dVector(mdVector)
    {}

    MatrixCuda(int _numRows, int _numCols, thrust::device_vector<T>&& mdVector): numRows(_numRows), numCols(_numCols), dVector(std::move(mdVector))
    {}
public:
    MatrixCuda(const MatrixCuda& matIn) = delete;
    MatrixCuda& operator=(const MatrixCuda& matIn) = delete;
    MatrixCuda(int _numRows, int _numCols): numRows(_numRows), numCols(_numCols), dVector(int64_t(numRows) * int64_t(numCols))
    {}

    MatrixCuda(const Matrix<T, majorOrder>& matHost): numRows(matHost.getNumRows()), numCols(matHost.getNumCols()),
        dVector(matHost.getDataC(), matHost.getDataC() + matHost.getNumElements())
    {}


    MatrixCuda(MatrixCuda&& matIn)
    {
        dVector = std::move(matIn.dVector);
        numRows = matIn.numRows;
        numCols = matIn.numCols;
    }

    MatrixCuda& operator=(MatrixCuda&& matIn)
    {
        dVector = std::move(matIn.dVector);
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

    int64_t getNumElements(void) const
    {
        return int64_t(numRows) * int64_t(numCols);
    }

    int64_t getSize(void) const
    {
        return int64_t(numRows) * int64_t(numCols) * sizeof(T);
    }

    int getLeadingDimension(void) const
    {
        if constexpr (majorOrder == MajorOrder::COL_MAJOR)
        {
            return getNumRows();
        }
        else
        {
            return getNumCols();
        }
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
        thrust::host_vector<T> hostVector = dVector;
        Matrix<T, majorOrder> matOut(hostVector.data(), getNumRows(), getNumCols());
        return matOut;
    }

    friend std::ostream& operator<<(std::ostream& out, MatrixCuda<T, majorOrder>& matCuda)
    {
        Matrix<T, majorOrder> matHost = matCuda.getHostCopy();

        out << matHost;

        return out;
    }

    MatrixCuda<T, majorOrder> copyMatrix(int startRowIdx, int endRowIdx, int startColIdx, int endColIdx) const;

    constexpr auto cudblasXgeam(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cublasSgeam;
        }
        else
        {
            return &cublasDgeam;
        }
    }


    auto changeLayout(void) const;

    static MatrixCuda<T, majorOrder> zero(int numRows, int numCols);
    static MatrixCuda<T, majorOrder> identity(int numRows);
    int computeInverse(MatrixCuda<T, majorOrder>& matInv) const;

    constexpr auto cublasXgemv(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cublasSgemv;
        }
        else
        {
            return &cublasDgemv;
        }
    }

    constexpr auto cublasXgemm(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cublasSgemm;
        }
        else
        {
            return &cublasDgemm;
        }
    }

    MatrixCuda<T, majorOrder> computeMatrixVector(const MatrixCuda<T, majorOrder>& vectV,
        bool transpose) const;

    MatrixCuda<T, majorOrder> multiply(const MatrixCuda<T, majorOrder>& mat2, int startRowIdx);

    MatrixCuda<T, majorOrder> selfMatrixTransposeMultiplication(void) const;



    constexpr auto cusolverDnXgeqrf_bufferSize(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cusolverDnSgeqrf_bufferSize;
        }
        else
        {
            return &cusolverDnDgeqrf_bufferSize;
        }
    }

    constexpr auto cusolverDnXgeqrf(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cusolverDnSgeqrf;
        }
        else
        {
            return &cusolverDnDgeqrf;
        }
    }

    constexpr auto cusolverDnXorgqr_bufferSize(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cusolverDnSorgqr_bufferSize;
        }
        else
        {
            return &cusolverDnDorgqr_bufferSize;
        }
    }

    constexpr auto cusolverDnXorgqr(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cusolverDnSorgqr;
        }
        else
        {
            return &cusolverDnDorgqr;
        }
    }

    int computeQRDecomposition(MatrixCuda<T, MajorOrder::COL_MAJOR>& matR,
        MatrixCuda<T, MajorOrder::COL_MAJOR>& matQ, bool computeQ = false,
        const std::string& memoryTag = "");

    MatrixCuda<T, majorOrder> solveLLSNormalEquationUsingR(
        const MatrixCuda<T, majorOrder>& matR,
        const MatrixCuda<T, majorOrder>& vectB) const;

    constexpr auto cusolverDnXgesvd_bufferSize(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cusolverDnSgesvd_bufferSize;
        }
        else
        {
            return &cusolverDnDgesvd_bufferSize;
        }
    }

    constexpr auto cusolverDnXgesvd(void) const
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return &cusolverDnSgesvd;
        }
        else
        {
            return &cusolverDnDgesvd;
        }
    }

    int computeSVDDecomposition(MatrixCuda<T, MajorOrder::COL_MAJOR>& matU,
        MatrixCuda<T, MajorOrder::COL_MAJOR>& matSigma, MatrixCuda<T, MajorOrder::COL_MAJOR>& matV,
        bool computeU = false, bool computeV = false, bool computeSigma = true);
};


template <typename T>
using MatrixCudaCol = MatrixCuda<T, MajorOrder::COL_MAJOR>;
template <typename T>
using MatrixCudaRow = MatrixCuda<T, MajorOrder::ROW_MAJOR>;

template <typename T, MajorOrder majorOrder>
__global__ void copyMatrixCuda(const T* d_ASrc, T* d_Bdst, int numRowsSrc, int numColsSrc, int startRowIdx, int endRowIdx, int startColIdx, int endColIdx) {
    int colIdxDst  = threadIdx.x;
    int rowIdxDst = blockIdx.x;
    int colIdxSrc = threadIdx.x + startColIdx;
    int rowIdxSrc = blockIdx.x + startRowIdx;
    int numRowsDst = endRowIdx - startRowIdx + 1;
    int numColsDst = endColIdx - startColIdx + 1;

    if (rowIdxSrc < numRowsSrc and colIdxSrc < numColsSrc)
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

// @note Number of columns should be smaller than 1024
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

// @note Number of columns should be smaller than 1024
template <typename T>
__global__ void setEyes(T* d_A, int numRows) {
    int colIdx = threadIdx.x;
    if (colIdx < numRows)
    {
        int posIdx = IDX_C(colIdx, colIdx, numRows, numRows);
        d_A[posIdx] = 1.0;
    }
}


template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder> MatrixCuda<T, majorOrder>::zero(int numRows, int numCols)
{
    MatrixCuda<T, majorOrder> outZeros {numRows, numCols,
        thrust::device_vector<double>(numRows * numCols, 0.0)};

    return outZeros;
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder> MatrixCuda<T, majorOrder>::identity(int numRows)
{
    MatrixCuda<T, majorOrder> outZeros{MatrixCuda<T, majorOrder>::zero(numRows, numRows)};
    setEyes<<<1, numRows>>>(outZeros.getData(), outZeros.getNumRows());

    return outZeros;
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder> MatrixCuda<T, majorOrder>::
copyMatrix(int startRowIdx, int endRowIdx, int startColIdx, int endColIdx) const
{
    MatrixCuda<T, majorOrder> matOut{endRowIdx - startRowIdx + 1, endColIdx - startColIdx + 1};
    copyMatrixCuda<T, majorOrder><<<matOut.getNumRows(), matOut.getNumCols()>>>
        (getDataC(), matOut.getData(), getNumRows(), getNumCols(), startRowIdx, endRowIdx, startColIdx, endColIdx);
    return matOut;
}

template <typename T, MajorOrder majorOrder>
int MatrixCuda<T, majorOrder>::computeInverse(MatrixCuda<T, majorOrder>& matInv) const
{
    matInv = copyMatrix(0, getNumRows() - 1, 0, getNumCols() -1);

    int *dPivots = nullptr;
    int *dInfo = nullptr;
    T *dWork = nullptr;
    int lwork = 0;

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));

    CUDA_CALL(cudaMalloc((void**)&dPivots, sizeof(int) * getNumRows()));
    CUDA_CALL(cudaMalloc((void**)&dInfo, sizeof(int)));
    CUSOLVER_CALL(cusolverDnDgetrf_bufferSize(cusolverH, getNumRows(), getNumRows(),
        matInv.getData(), matInv.getLeadingDimension(), &lwork));
    CUDA_CALL(cudaMalloc((void**)&dWork, sizeof(T) * lwork));

    // LU Decomposition
    CUSOLVER_CALL(cusolverDnDgetrf(cusolverH, matInv.getNumRows(), matInv.getNumRows(),
        matInv.getData(), matInv.getLeadingDimension(), dWork, dPivots, dInfo));

    auto matIdentity = MatrixCuda<T, majorOrder>::identity(matInv.getNumRows());

    CUSOLVER_CALL(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, matInv.getNumRows(),
        matInv.getNumRows(), matInv.getData(), matInv.getLeadingDimension(),
        dPivots, matIdentity.getData(), matIdentity.getNumRows(), dInfo));
    matInv = std::move(matIdentity);
    CUDA_CALL(cudaFree(dPivots));
    CUDA_CALL(cudaFree(dInfo));
    CUDA_CALL(cudaFree(dWork));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));

    return 0;
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder>
MatrixCuda<T, majorOrder>::computeMatrixVector(const MatrixCuda<T, majorOrder>& vectV,
        bool transpose) const
{
    cublasHandle_t handle;
    CUBLASS_CALL(cublasCreate(&handle));

    T alpha = 1.0;
    T beta = 0.0;
    auto transposeLay = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto rowsOut = (transpose ? getNumCols() : getNumRows());
    MatrixCuda<T, majorOrder> matOut{rowsOut, 1};

    CUBLASS_CALL(cublasXgemv()(
        handle,
        transposeLay,
        getNumRows(),
        getNumCols(),
        &alpha,
        getDataC(), getLeadingDimension(),
        vectV.getDataC(), 1,
        &beta,
        matOut.getData(), 1
    ));

    CUBLASS_CALL(cublasDestroy(handle));

    return matOut;
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder> MatrixCuda<T, majorOrder>::multiply(
    const MatrixCuda<T, majorOrder>& mat2, int startRowIdx = 0)
{
    cublasHandle_t handle;
    CUBLASS_CALL(cublasCreate(&handle));

    double alpha = 1.0;
    double beta = 0.0;
    MatrixCuda<T, majorOrder> matOut{getNumRows(), mat2.getNumCols()};

    if constexpr (majorOrder == MajorOrder::COL_MAJOR)
    {
        CUBLASS_CALL(cublasXgemm()(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            getNumRows(), mat2.getNumCols(),
            getNumCols(), &alpha, getDataC(), getLeadingDimension(),
            startRowIdx + mat2.getDataC(), mat2.getLeadingDimension(), &beta,
            matOut.getData(), matOut.getLeadingDimension()));
    }
    else
    {
        CUBLASS_CALL(cublasXgemm()(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            mat2.getNumCols(), getNumRows(),
            getNumCols(), &alpha,
            mat2.getDataC() + startRowIdx * mat2.getNumCols(),
            mat2.getLeadingDimension(),
            getDataC(), getLeadingDimension(),
            &beta,
            matOut.getData(), matOut.getLeadingDimension()));
    }

    CUBLASS_CALL(cublasDestroy(handle));
    return matOut;
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder>
MatrixCuda<T, majorOrder>::selfMatrixTransposeMultiplication(void) const
{
    T alpha = 1.0;
    T beta = 0.0;

    cublasHandle_t handle;
    CUBLASS_CALL(cublasCreate(&handle));
    MatrixCuda<T, majorOrder> matOut{getNumRows(), getNumRows()};

    // C = A * Aᵀ
    // A is m x n, Aᵀ is n x m ⇒ C is m x m
    // Use: C = alpha * A * Aᵀ + beta * C
    // Note: cuBLAS is column-major by default, so interpret as:
    // C = A * Aᵀ ⇒ C = op(A) * op(B) with op(A) = N, op(B) = T
    CUBLASS_CALL(cublasXgemm()(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        getNumRows(),
        getNumRows(),
        getNumCols(),
        &alpha,
        getDataC(), getLeadingDimension(),
        getDataC(), getLeadingDimension(),
        &beta,
        matOut.getData(), matOut.getLeadingDimension()
    ));

    CUBLASS_CALL(cublasDestroy(handle));

    return matOut;
}



template <typename T, MajorOrder majorOrder>
int MatrixCuda<T, majorOrder>::computeQRDecomposition(MatrixCudaCol<T>& matR,
        MatrixCuda<T, MajorOrder::COL_MAJOR>& matQ, bool computeQ,
        const std::string& memoryTag)
{
    auto memUsed = getCudaMemoryUsage();
    if (majorOrder == MajorOrder::COL_MAJOR)
    {
        cusolverDnHandle_t cuSolverHandle;
        CUSOLVER_CALL(cusolverDnCreate(&cuSolverHandle));
        int workspace_size = 0;

        CUSOLVER_CALL(cusolverDnXgeqrf_bufferSize()(cuSolverHandle, getNumRows(), getNumCols(), getData(), getNumRows(), &workspace_size));

        // Allocate workspace
        T *dWork, *dTau;
        MEMORY_LOG(memoryTag, "Memory after allocating all QR cuSolver");
        CUDA_CALL(cudaMalloc((void**)&dWork, workspace_size * sizeof(T)));

        // Allocate device status variable
        int *devInfo;
        CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)));
        CUDA_CALL(cudaMalloc((void**)&dTau, std::min(getNumRows(), getNumCols()) * sizeof(T)));
        MEMORY_LOG(memoryTag, "Memory after allocating all QR cuSolver buffers");
        CUSOLVER_CALL(cusolverDnXgeqrf()(cuSolverHandle, getNumRows(), getNumCols(), getData(), getLeadingDimension(), dTau, dWork, workspace_size, devInfo));

        matR = copyMatrix(0, getNumCols() - 1, 0, getNumCols() - 1);
        setZerosUpperTriangularCol<<<matR.getNumRows(), matR.getNumCols()>>>(matR.getData(), matR.getNumRows(), matR.getNumCols());
        MEMORY_LOG(memoryTag, "Memory after allocating R");
        if (computeQ)
        {
            matQ = copyMatrix(0, getNumRows() - 1, 0, getNumCols() - 1);
            MEMORY_LOG(memoryTag, "Memory after allocating Q");
            CUSOLVER_CALL(cusolverDnXorgqr_bufferSize()(cuSolverHandle, getNumRows(), getNumCols(),
             getNumCols(), getData(), getLeadingDimension(), dTau, &workspace_size));
            CUDA_CALL(cudaFree(dWork));

            CUDA_CALL(cudaMalloc((void**)&dWork, sizeof(T) * workspace_size));

            CUSOLVER_CALL(cusolverDnXorgqr()(cuSolverHandle, matQ.getNumRows(), matQ.getNumCols(),
                matQ.getNumCols(), matQ.getData(), matQ.getLeadingDimension(), dTau, dWork,
                    workspace_size, devInfo));
        }

        CUDA_CALL(cudaFree(dTau));
        CUDA_CALL(cudaFree(dWork));
        CUDA_CALL(cudaFree(devInfo));
        CUSOLVER_CALL(cusolverDnDestroy(cuSolverHandle));
    }
    return 0;
}

template <typename T, MajorOrder majorOrder>
MatrixCuda<T, majorOrder>
MatrixCuda<T, majorOrder>::solveLLSNormalEquationUsingR(
    const MatrixCuda<T, majorOrder>& matR, const MatrixCuda<T, majorOrder>& vectB) const
{
    MatrixCuda<T, majorOrder> matRInv{1, 1};
    matR.computeInverse(matRInv);
    auto outMat = matRInv.selfMatrixTransposeMultiplication();

    auto tempVect = this->computeMatrixVector(vectB, true);
    auto vectXOut = outMat.computeMatrixVector(tempVect, false);
    return vectXOut;
}

template <typename T, MajorOrder majorOrder>
int MatrixCuda<T, majorOrder>::computeSVDDecomposition(
    MatrixCuda<T, MajorOrder::COL_MAJOR>& matU, MatrixCuda<T, MajorOrder::COL_MAJOR>& matSigma, MatrixCuda<T, MajorOrder::COL_MAJOR>& matVT,
    bool computeU, bool computeV, bool computeSigma)
{
    char jobu;  // No computation of U
    char jobvt; // No computation of V^T
    if (not computeU and not computeV and computeSigma)
    {
        jobu = 'N';
        jobvt = 'N';
    }
    else
    {
        jobu = 'S';
        jobvt = 'S';
    }

    auto ldA = getLeadingDimension();
    int *dInfo;
    T *dWork;
    int lwork = 0;

    int rank = std::min(getNumRows(), getNumCols());
    matSigma = std::move(Matrix<T, majorOrder>(rank, 1));

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));
    CUDA_CALL(cudaMalloc((void**)&dInfo, sizeof(int)));
    CUSOLVER_CALL(cusolverDnXgesvd_bufferSize()(cusolverH, getNumRows(), getNumCols(), &lwork));
    CUDA_CALL(cudaMalloc((void**)&dWork, sizeof(T) * lwork));

    CUSOLVER_CALL(cusolverDnXgesvd()(cusolverH, jobu, jobvt,
        getNumRows(), getNumCols(), getData(), ldA,
        matSigma.getData(), matU.getData(), matU.getLeadingDimension(),
            matVT.getData(), matVT.getLeadingDimension(), dWork, lwork, nullptr, dInfo));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));
    CUDA_CALL(cudaFree(dInfo));
    CUDA_CALL(cudaFree(dWork));

    return 0;
}

template<typename T, MajorOrder majorOrder>
auto MatrixCuda<T, majorOrder>::changeLayout(void) const
{
    using MatrixOpositeLayoutType = typename std::conditional<
        majorOrder == MajorOrder::ROW_MAJOR,
        MatrixCudaCol<T>, MatrixCudaRow<T>>::type;

    MatrixOpositeLayoutType matCudaChgLayout(getNumRows(), getNumCols());

    cublasHandle_t handle;
    CUBLASS_CALL(cublasCreate(&handle));

    const T alpha = 1.0f;
    const T beta = 0.0f;

    auto cudblasXgeamFun = cudblasXgeam();

    CUBLASS_CALL(cudblasXgeamFun(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        getNumRows(), getNumCols(),
        &alpha,
        getDataC(), getNumCols(),
        &beta,
        nullptr, getNumCols(),
        matCudaChgLayout.getData(), matCudaChgLayout.getNumRows()));
    CUBLASS_CALL(cublasDestroy(handle));

    return matCudaChgLayout;
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