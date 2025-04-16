#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <random>
#include <iomanip>
#include "matrix.h"
#include "matrix_cuda.h"

__device__ double sqrt(double x);

// We assume there are less than 1024 columns,
// otherwise this should be updated
template<typename T>
__global__ void computeHeadsAndTails(T* dMat, const int* dOffsets, const int* dJoinSizes, int numCols) {
    T dataHeadsL;
    int colIdx = threadIdx.x;
    int offsetIdx = blockIdx.x;
    int headRowIdx = dOffsets[offsetIdx];
    int numRows = dJoinSizes[offsetIdx];

    if (colIdx == 0 or colIdx >= numCols)
    {
        return;
    }
    dataHeadsL = dMat[IDX_R(headRowIdx, colIdx, numRows, numCols)];
    for (int rowIdx = headRowIdx + 1; rowIdx < headRowIdx + numRows; rowIdx++)
    {
        T i = rowIdx - headRowIdx + 1;

        T prevRowSum;
        T tailVal;
        prevRowSum = dataHeadsL;
        T matVal = dMat[IDX_R(rowIdx, colIdx, numRows, numCols)];
        dataHeadsL += matVal;
        tailVal = (matVal * (i - 1) - prevRowSum) / sqrt(i * (i - 1));
        dMat[IDX_R(rowIdx, colIdx, numRows, numCols)] = tailVal;
    }
    dMat[IDX_R(headRowIdx, colIdx, numRows, numCols)] = dataHeadsL / sqrt((double)numRows);
}

// We assume there are less than 1024 columns,
// otherwise this should be updated
template <typename T>
__global__ void concatenateHeadsAndTails(const T* dMat, const T* dMat2Mod, T* dOutMat, const int* dNumRows1, int numCols1,
        const int* dNumRows2, int numCols2, const int* dOffsets1, const int* dOffsets2, const int* dOffsets) {
    int colIdx = threadIdx.x;
    int offsetIdx = blockIdx.x;
    int headRowIdx1 = dOffsets1[offsetIdx];
    int headRowIdx2 = dOffsets2[offsetIdx];
    int headRowsIdxOut = dOffsets[offsetIdx];
    int numRows1 = dNumRows1[offsetIdx];
    int numRows2 = dNumRows2[offsetIdx];
    const int numRowsOut = numRows1 + numRows2 - 1;
    const int numColsOut = numCols1 + numCols2 - 1;

    for (int rowIdx1 = headRowIdx1; rowIdx1 < headRowIdx1 + numRows1; rowIdx1++)
    {
        int outRowIdx = rowIdx1 - headRowIdx1 + headRowsIdxOut;
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(outRowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = dMat[IDX_R(rowIdx1, colIdx, numRows1, numCols1)] * sqrt((double)numRows2);
        }
        if (colIdx > 0 and colIdx < numCols2)
        {
            int posIdx2 = IDX_R(outRowIdx, colIdx + numCols1 - 1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = dMat2Mod[IDX_R(headRowIdx2, colIdx, numRows2, numCols2)];
        }
    }
    for (int rowIdx2 = headRowIdx2; rowIdx2 < headRowIdx2 + numRowsOut - numRows1; rowIdx2++)
    {
        int outRowIdx = rowIdx2 - headRowIdx2 + numRows1 + headRowsIdxOut;
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(outRowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = 0;
        }
        if (colIdx > 0 and colIdx < numCols2)
        {
            int posIdx2 = IDX_R(outRowIdx, colIdx + numCols1 - 1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = dMat2Mod[IDX_R(rowIdx2 + 1, colIdx, numRows2, numCols2)] * sqrt((double)numRows1);
        }
    }
}

// TODO: Add inverse, matrix matrix multiplication

template <typename T>
__global__ void findUniqueOffsets(const T* d_arr, int* dDiffPrevRow, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    if (idx == 0 || (d_arr[idx] != d_arr[idx - 1]))
    {
        dDiffPrevRow[idx] = idx;
    }
    else
    {
        dDiffPrevRow[idx] = -1;
    }
}

template <typename T>
__global__ void joinAdd(const T* dMat1, const T* dMat2, T* dOutMat,
    int numCols, const int* dOffsets1, const int* dOffsets2, const int* dOffsets,
    const int* dJoinSizes1, const int* dJoinSizes2, const int* dJoinSizes)
{
    int colIdx = threadIdx.x;
    int idxDist = blockIdx.x;
    int startRowIdx1 = dOffsets1[idxDist];
    int startRowIdx2 = dOffsets2[idxDist];
    int endRowIdx1 = dOffsets1[idxDist+1];
    int endRowIdx2 = dOffsets2[idxDist+1];
    int outRowIdx = dOffsets[idxDist];

    int joinSize1 = dJoinSizes1[idxDist];
    int joinSize2 = dJoinSizes2[idxDist];
    int joinSize = dJoinSizes[idxDist];
    if (colIdx >= numCols)
    {
        return;
    }
    for (int rowIdx1 = startRowIdx1; rowIdx1 < endRowIdx1; rowIdx1++)
    {
        for (int rowIdx2 = startRowIdx2; rowIdx2 < endRowIdx2; rowIdx2++)
        {
            int posIdx = IDX_R(outRowIdx, colIdx, joinSize, numCols);
            int posIdx1 = IDX_R(rowIdx1, colIdx, joinSize1, numCols);
            int posIdx2 = IDX_R(rowIdx2, colIdx, joinSize2, numCols);
            dOutMat[posIdx] = dMat1[posIdx1] + dMat2[posIdx2];
            outRowIdx++;
        }
    }
}
template <typename T>
void computeOffsets(const MatrixCudaRow<T>& matA, thrust::device_vector<int>& dOffsets)
{
    int numRows = matA.getNumRows();
    thrust::device_vector<int> dDiffPrevRow(numRows);

    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;
    auto matAJoinCol = matA.getColumn(0);
    findUniqueOffsets<<<numBlocks, blockSize>>>(matAJoinCol.getDataC(), dDiffPrevRow.data().get(), numRows);

    int countDiff = numRows - thrust::count(dDiffPrevRow.begin(), dDiffPrevRow.end(), -1);
    dOffsets = std::move(thrust::device_vector<int>(countDiff + 1));
    thrust::copy_if(dDiffPrevRow.begin(), dDiffPrevRow.end(), dOffsets.begin(), [] __device__ (int val) { return val != -1; });
    dOffsets.back() = matA.getNumRows();
}

void computeJoinSizes(const thrust::device_vector<int>& dOffsets, thrust::device_vector<int>& dJoinSizes)
{
    dJoinSizes = std::move(thrust::device_vector<int>(dOffsets.size() - 1));

    auto first = thrust::make_zip_iterator(thrust::make_tuple(dOffsets.begin(), dOffsets.begin() + 1));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(dOffsets.end() - 1, dOffsets.end()));

    thrust::transform(first, last, dJoinSizes.begin(), [] __device__ (const thrust::tuple<int, int>& x) { return thrust::get<1>(x) - thrust::get<0>(x); });
}

void computeJoinSizeOfTwoTables(const thrust::device_vector<int>& dJoinSizes1, const thrust::device_vector<int>& dJoinSizes2,
    thrust::device_vector<int>& dJoinSize, thrust::device_vector<int>& dJoinOffsets)
{
    dJoinSize = std::move(thrust::device_vector<int>(dJoinSizes1.size()));
    dJoinOffsets = std::move(thrust::device_vector<int>(dJoinSizes1.size() + 1));

    thrust::transform(dJoinSizes1.begin(), dJoinSizes1.end(), dJoinSizes2.begin(), dJoinSize.begin(), thrust::multiplies<int>());
    dJoinOffsets.front() = 0;
    thrust::inclusive_scan(dJoinSize.begin(), dJoinSize.end(), dJoinOffsets.begin() + 1);
}

void computeFigaroOutputOffsetsTwoTables(const thrust::device_vector<int>& dJoinSizes1, const thrust::device_vector<int>& dJoinSizes2,
    thrust::device_vector<int>& dJoinSizes, thrust::device_vector<int>& dJoinOffsets)
{
    dJoinSizes = std::move(thrust::device_vector<int>(dJoinSizes1.size()));
    dJoinOffsets = std::move(thrust::device_vector<int>(dJoinSizes1.size() + 1));

    thrust::transform(dJoinSizes1.begin(), dJoinSizes1.end(), dJoinSizes2.begin(), dJoinSizes.begin(), thrust::plus<int>());
    thrust::transform(dJoinSizes.begin(), dJoinSizes.end(), dJoinSizes.begin(),  [] __device__ (int x) { return x - 1; });

    dJoinOffsets.front() = 0;
    thrust::inclusive_scan(dJoinSizes.begin(), dJoinSizes.end(), dJoinOffsets.begin() + 1);
}

template <typename T>
void computeHeadsAndTails(MatrixCudaRow<T>& matCuda, const thrust::device_vector<int>& dOffsets, const thrust::device_vector<int>& dJoinSizes)
{
    computeHeadsAndTails<<<dJoinSizes.size(), matCuda.getNumCols()>>>
        (matCuda.getData(), dOffsets.data().get(), dJoinSizes.data().get(), matCuda.getNumCols());
}

template <typename T>
void concatenateHeadsAndTails(const MatrixCudaRow<T>& matCuda1, const MatrixCudaRow<T>& matCuda2, MatrixCudaRow<T>& matCudaOut,
    const thrust::device_vector<int>& dJoinSizes1, const thrust::device_vector<int>& dJoinSizes2, const thrust::device_vector<int>& dOffsets1,
        const thrust::device_vector<int>& dOffsets2, const thrust::device_vector<int>& dJoinOffsets)
{
    int numRows1 = matCuda1.getNumRows();
    int numCols1 = matCuda1.getNumCols();
    int numRows2 = matCuda2.getNumRows();
    int numCols2 = matCuda2.getNumCols();
    concatenateHeadsAndTails<<<dJoinSizes1.size(), max(numCols1, numCols2)>>>(
        matCuda1.getDataC(), matCuda2.getDataC(), matCudaOut.getData(),
        dJoinSizes1.data().get(), numCols1, dJoinSizes2.data().get(), numCols2,
        dOffsets1.data().get(), dOffsets2.data().get(), dJoinOffsets.data().get());
}

template <typename T>
void joinAdd(const MatrixCudaRow<T>& matCuda1, const MatrixCudaRow<T>& matCuda2,
    MatrixCudaRow<T>& matCudaOut,
    const thrust::device_vector<int>& dOffsets1,
    const thrust::device_vector<int>& dOffsets2,
    const thrust::device_vector<int>& dOffsets,
    const thrust::device_vector<int>& dJoinSizes1,
    const thrust::device_vector<int>& dJoinSizes2,
    const thrust::device_vector<int>& dJoinSizes)
{
    joinAdd<<<dJoinSizes1.size(), matCudaOut.getNumCols()>>>(
        matCuda1.getDataC(), matCuda2.getDataC(), matCudaOut.getData(),
        matCudaOut.getNumCols(),
        dOffsets1.data().get(), dOffsets2.data().get(),
        dOffsets.data().get(),  dJoinSizes1.data().get(),
        dJoinSizes2.data().get(), dJoinSizes.data().get());
}

template <typename T>
int computeFigaro(const MatrixRow<T>& mat1, const MatrixRow<T>& mat2,
    MatrixCol<T>& matR, MatrixCol<T>& matQ,  MatrixCol<T>& matU,
    MatrixCol<T>& matSigma, MatrixCol<T>& matV, ComputeDecomp decompType)
{
    auto memUsed = getCudaMemoryUsage();
    MEMORY_LOG("LinScale", "Memory at beginning LinScale")
    MatrixCudaRow<T> matCuda1(mat1), matCuda2(mat2);
    thrust::device_vector<int> dOffsets1, dOffsets2;

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    CUDA_CALL(cudaEventRecord(start));

    auto matCuda2NonJoin = matCuda2.copyMatrix(0, matCuda2.getNumRows() - 1,
        1, matCuda2.getNumCols() - 1);
    computeOffsets(matCuda1, dOffsets1);
    computeOffsets(matCuda2, dOffsets2);

    thrust::device_vector<int> dJoinSizes1, dJoinSizes2, dJoinSizes, dJoinOffsets;
    computeJoinSizes(dOffsets1, dJoinSizes1);
    computeJoinSizes(dOffsets2, dJoinSizes2);
    computeFigaroOutputOffsetsTwoTables(dJoinSizes1, dJoinSizes2, dJoinSizes, dJoinOffsets);

    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY or
        decompType == ComputeDecomp::U_AND_S_AND_V;
    bool computeQ = decompType == ComputeDecomp::Q_AND_R;
    bool computeU = decompType == ComputeDecomp::U_AND_S_AND_V;
    bool computeV = decompType == ComputeDecomp::U_AND_S_AND_V;

    int numRowsOut{dJoinOffsets.back()}, numColsOut{mat1.getNumCols() + mat2.getNumCols() - 1};
    MatrixCudaRow<T> matCudaOut(numRowsOut, numColsOut);

    computeHeadsAndTails(matCuda2, dOffsets2, dJoinSizes2);
    concatenateHeadsAndTails(matCuda1, matCuda2, matCudaOut, dJoinSizes1, dJoinSizes2, dOffsets1, dOffsets2, dJoinOffsets);
    auto matCudaTran = matCudaOut.changeLayout();

    int rank = min(numRowsOut, numColsOut);

    MatrixCudaCol<T> matRCuda{1, 1}, matQCuda{1, 1}, matUCuda{1, 1}, matSigmaCuda{1, 1}, matVCuda{1, 1};
    MatrixCudaRow<T> matQRowCuda{1, 1};
    MEMORY_LOG("LinScale", "Memory at the end of LinScale")
    matCudaTran.computeQRDecomposition(matRCuda, matQCuda, false, "LinScale");

    if (computeSVD)
    {

        matRCuda.computeSVDDecomposition(matUCuda, matSigmaCuda, matVCuda,
                computeU, computeV, true);
        if (computeU)
        {
            // inverse...
            // TODO: Copy logic from Q
        }
    }
    else
    {
        if (computeQ)
        {
            MatrixCudaCol<T> matRInvCuda{1, 1};
            matRCuda.computeInverse(matRInvCuda);
            auto matRInvRowCuda = matRInvCuda.changeLayout();
            auto matCudaRow1MulRInv =  matCuda1.multiply(matRInvRowCuda, 0);
            auto matCudaRow2MulRInv =  matCuda2NonJoin.multiply(matRInvRowCuda, matCuda1.getNumCols());
            thrust::transform(dJoinSizes1.begin(), dJoinSizes1.end(), dJoinSizes2.begin(), dJoinSizes.begin(), thrust::multiplies<float>());
            dJoinOffsets.front() = 0;
            thrust::inclusive_scan(dJoinSizes.begin(), dJoinSizes.end(), dJoinOffsets.begin() + 1);

            matQRowCuda = std::move(MatrixCudaRow<T>{dJoinOffsets.back(), matRInvRowCuda.getNumCols()});
            MEMORY_LOG("LinScale", "After allocating all structures")
            joinAdd(matCudaRow1MulRInv, matCudaRow2MulRInv, matQRowCuda, dOffsets1, dOffsets2,
                dJoinOffsets, dJoinSizes1, dJoinSizes2, dJoinSizes);

        }
    }

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (computeSVD)
    {
        matSigma = matSigmaCuda.getHostCopy();
        if (computeU)
        {
            matU = matUCuda.getHostCopy();
            matV = matVCuda.getHostCopy();
        }
    }
    else
    {
        matR = matRCuda.getHostCopy();
        if (computeQ)
        {
            matQCuda = matQRowCuda.changeLayout();
            matQ = matQCuda.getHostCopy();
        }
    }

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    std::cout << "\n";
    if (computeSVD)
    {
	    std::cout << "SVD decomposition ";
    }
    else
    {
	    std::cout << "QR decomposition ";
    }
    std::cout << "Linscale took " << milliseconds << " ms.\n";

    return 0;
}

template <typename T, MajorOrder majorOrder>
int computeGeneral(const Matrix<T, majorOrder>& matA,
    MatrixCol<T>& matR, MatrixCol<T>& matQ, MatrixCol<T>& matU,
    MatrixCol<T>& matSigma, MatrixCol<T>& matV, ComputeDecomp decompType)
{
    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY or decompType == ComputeDecomp::U_AND_S_AND_V;
    bool computeQ = decompType == ComputeDecomp::Q_AND_R;
    bool computeU = decompType == ComputeDecomp::U_AND_S_AND_V;
    bool computeV = decompType == ComputeDecomp::U_AND_S_AND_V;
    MEMORY_LOG("CUDA", "Memory at the beginning cuSolver")
    MatrixCudaCol<T> matACuda(matA);
    MatrixCudaCol<T> matACudaCol{1, 1};

    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        matACudaCol = matACuda.changeLayout();
    }
    else
    {
        matACudaCol = std::move(matACuda);
    }

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));
    MatrixCudaCol<T> matRCuda{1, 1}, matQCuda{1, 1}, matUCuda{1, 1}, matSigmaCuda{1, 1}, matVCuda{1, 1};
    if (computeSVD)
    {
        matACudaCol.computeSVDDecomposition(matUCuda, matSigmaCuda, matVCuda,
            computeU, computeV, true);
    }
    else
    {
        matACudaCol.computeQRDecomposition(matRCuda, matQCuda, computeQ, "CUDA");
    }
    MEMORY_LOG("CUDA", "Memory at the end cuSolver")

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (computeSVD)
    {
        matSigma = matSigmaCuda.getHostCopy();
        if (computeU)
        {
            matU = matUCuda.getHostCopy();
            matV = matVCuda.getHostCopy();
        }
    }
    else
    {
        matR = matRCuda.getHostCopy();
        if (computeQ)
        {
            MEMORY_LOG("CUDA", "Memory before host copy")
            matQ = matQCuda.getHostCopy();
            MEMORY_LOG("CUDA", "Memory after host copy")
        }
    }

    std::string nameDecomp = computeSVD ? "SVD" : "QR";
    std::cout << "\n" + nameDecomp + " decomposition CUSolver took " << milliseconds << " ms.\n";

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    return 0;
}

template <typename T, MajorOrder majorOrder>
int solveLLSNormalEquationUsingR(const Matrix<T, majorOrder>& matA,
    const Matrix<T, majorOrder>& matR,
    const Matrix<T, majorOrder>& vectB,
    Matrix<T, majorOrder>& vectX)
{
    MatrixCuda<T, majorOrder> matACuda(matA);
    MatrixCuda<T, majorOrder> matRCuda(matR);
    MatrixCuda<T, majorOrder> vectBCuda(vectB);
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    auto matXCuda = matACuda.solveLLSNormalEquationUsingR(matR, vectB);
    vectX = matXCuda.getHostCopy();

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
    // std::cout << "LLS using normal equations and R took " << milliseconds << " ms.\n";

    return 0;
}

// template int computeGeneral<double, MajorOrder::ROW_MAJOR>(const MatrixDRow& matA,
//     MatrixDCol& matR, const std::string& fileName, ComputeDecomp decompType);

// template int computeGeneral<double, MajorOrder::COL_MAJOR>(const MatrixDCol& matA,
//         MatrixDCol& matR, MatrixDCol& matQ, ComputeDecomp decompType);

template int computeGeneral<double, MajorOrder::COL_MAJOR>(const MatrixDCol& matA,
    MatrixDCol& matR, MatrixDCol& matQ, MatrixDCol& matU,
    MatrixDCol& matSigma, MatrixDCol& matV, ComputeDecomp decompType);


template int computeFigaro<double>(const MatrixDRow& mat1, const MatrixDRow& mat2,
            MatrixDCol& matR, MatrixDCol& matQ,  MatrixDCol& matU,
            MatrixDCol& matSigma, MatrixDCol& matV, ComputeDecomp decompType);

// template int computeFigaro<double>(const MatrixDRow& mat1, const MatrixDRow& mat2,
//     MatrixDCol& matR, MatrixDCol& matQ, ComputeDecomp decompType);


template int solveLLSNormalEquationUsingR<double, MajorOrder::COL_MAJOR>(
    const MatrixDCol& matA, const MatrixDCol& matR,
    const MatrixDCol& vectB, MatrixDCol& vectX);