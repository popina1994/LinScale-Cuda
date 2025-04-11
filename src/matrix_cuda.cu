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
int computeFigaro(const MatrixRow<T>& mat1, const MatrixRow<T>& mat2,
    MatrixCol<T>& matR, MatrixCol<T>& matQ, const std::string& fileName, ComputeDecomp decompType)
{
    MatrixCudaRow<T> matCuda1(mat1), matCuda2(mat2);
    thrust::device_vector<int> dOffsets1, dOffsets2;

    // std::cout << "C1" << matCuda1;
    computeOffsets(matCuda1, dOffsets1);
    // std::cout << "Offsets 1" << std::endl;
    // printDeviceVector(dOffsets1);
    // std::cout << "C2" << matCuda2;
    computeOffsets(matCuda2, dOffsets2);
    // std::cout << "Offsets 2" << std::endl;
    // printDeviceVector(dOffsets2);

    thrust::device_vector<int> dJoinSizes1, dJoinSizes2, dJoinSizes, dJoinOffsets;
    computeJoinSizes(dOffsets1, dJoinSizes1);
    computeJoinSizes(dOffsets2, dJoinSizes2);
    computeFigaroOutputOffsetsTwoTables(dJoinSizes1, dJoinSizes2, dJoinSizes, dJoinOffsets);

    // std::cout << "JOIN SIZES 1" << std::endl;
    // printDeviceVector(dJoinSizes1);
    // std::cout << "JOIN SIZES 2" << std::endl;
    // printDeviceVector(dJoinSizes2);
    // std::cout << "JOIN SIZES" << std::endl;
    // printDeviceVector(dJoinSizes);
    // std::cout << "JOIN OFFSETS" << std::endl;
    // printDeviceVector(dJoinOffsets);

    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY;

    int numRowsOut{dJoinOffsets.back()}, numColsOut{mat1.getNumCols() + mat2.getNumCols() - 1};
    MatrixCudaRow<T> matCudaOut(numRowsOut, numColsOut);

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // Start measuring time
    CUDA_CALL(cudaEventRecord(start));

    computeHeadsAndTails(matCuda2, dOffsets2, dJoinSizes2);
    // std::cout << "C2 Modified" << matCuda2;
    concatenateHeadsAndTails(matCuda1, matCuda2, matCudaOut, dJoinSizes1, dJoinSizes2, dOffsets1, dOffsets2, dJoinOffsets);
    auto matCudaTran = changeLayoutFromRowToColumn(matCudaOut);

    int rank = min(numRowsOut, numColsOut);
    // Compute QR factorization
    MatrixCudaCol<T> matRCuda{1, 1}, matQCuda{1, 1}, matUCuda{1, 1}, matSigmaCuda{1, 1}, matVCuda{1, 1};
    matCudaTran.computeQRDecomposition(matRCuda, matQCuda, false);
    // std::cout << matRCuda << std::endl;
    if (computeSVD)
    {
        matRCuda.computeSVDDecomposition(matUCuda, matSigmaCuda, matVCuda);
        // computeQ
        // compute the inverse of the R
        // compute the multiplication of the inverse by both matrices (excluding join columns)
        // join values
    }

    // Stop measuring time
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (computeSVD)
    {
        auto matSigma = matSigmaCuda.getHostCopy();
        printMatrix(matSigma, matSigma.getNumRows(), fileName + "LinScaleS", false);
    }
    else
    {
        matR = matRCuda.getHostCopy();
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
int computeGeneral(const Matrix<T, majorOrder>& matA, MatrixCol<T>& matR,
    MatrixCol<T>& matQ, const std::string& fileName, ComputeDecomp decompType)
{
    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY;
    bool computeQ = decompType == ComputeDecomp::Q_AND_R;

    MatrixCudaCol<T> matACuda(matA);
    MatrixCudaCol<T> matACudaCol{1, 1};

    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        matACudaCol = changeLayoutFromRowToColumn(matACuda);
    }
    else
    {
        matACudaCol = std::move(matACuda);
    }

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));
    MatrixCudaCol<T> matRCuda{1, 1}, matQCuda{1, 1}, matU{1, 1}, matSigmaCuda{1, 1}, matV{1, 1};

    if (computeSVD)
    {
        matACudaCol.computeSVDDecomposition(matU, matSigmaCuda, matV);
    }
    else
    {
        matACudaCol.computeQRDecomposition(matRCuda, matQCuda, computeQ);
        // computeQ
        // compute the inverse of the R
        // compute the multiplication of the inverse by both matrices (excluding join columns)
        // join values
    }

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results back to host
    if (computeSVD)
    {
        auto matSigma = matSigmaCuda.getHostCopy();
        printMatrix(matSigma, matSigma.getNumRows(), fileName + "cuSolverS", false);
    }
    else
    {
        matR = matRCuda.getHostCopy();
        if (computeQ)
        {
            matQ = matQCuda.getHostCopy();
        }
    }

    // Print execution time
    std::string nameDecomp = computeSVD ? "SVD" : "QR";
    std::cout << "\n" + nameDecomp + " decomposition CUSolver took " << milliseconds << " ms.\n";

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    return 0;
}

// template int computeGeneral<double, MajorOrder::ROW_MAJOR>(const MatrixDRow& matA,
//     MatrixDCol& matR, const std::string& fileName, ComputeDecomp decompType);

template int computeGeneral<double, MajorOrder::COL_MAJOR>(const MatrixDCol& matA,
        MatrixDCol& matR, MatrixDCol& matQ, const std::string& fileName, ComputeDecomp decompType);

template int computeFigaro<double>(const MatrixDRow& mat1, const MatrixDRow& mat2,
    MatrixDCol& matR, MatrixDCol& matQ, const std::string& fileName, ComputeDecomp decompType);