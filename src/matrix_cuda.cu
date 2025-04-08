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

// CUDA error check macro
#define CUDA_CALL(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
            return EXIT_FAILURE; \
        } \
    } while (0)


// cuSOLVER error check macro
#define CUSOLVER_CALL(call) \
    do { \
        cusolverStatus_t err = call; \
        if (err != CUSOLVER_STATUS_SUCCESS) { \
            std::cerr << "cuSOLVER Error: " << err << " at " << __LINE__ << std::endl; \
            return EXIT_FAILURE; \
        } \
    } while (0)


__device__ double sqrt(double x);

template<typename T>
__global__ void computeHeadsAndTails(T* d_mat, int numRows, int numCols) {
    extern __shared__ T dataHeads  [];
    int colIdx = threadIdx.x;
    int headRowIdx = 0;

    if (colIdx < numCols)
    {
        dataHeads[colIdx] = d_mat[IDX_R(headRowIdx, colIdx, numRows, numCols)];
    }
    __syncthreads();
    for (int rowIdx = headRowIdx + 1; rowIdx < numRows; rowIdx++)
    {
        T i = rowIdx - headRowIdx + 1;
        if (colIdx < numCols)
        {
            T prevRowSum;
            T tailVal;
            prevRowSum = dataHeads[colIdx];
            T matVal = d_mat[IDX_R(rowIdx, colIdx, numRows, numCols)];
            dataHeads[colIdx] += matVal;
            tailVal = (matVal * (i - 1) - prevRowSum) / sqrt(i * (i - 1));
            d_mat[IDX_R(rowIdx, colIdx, numRows, numCols)] = tailVal;
            // printf("TAIL VAL %d %d %.3f %.3f\n", rowIdx, colIdx, i, tailVal);
        }
        __syncthreads();
    }
    if (colIdx < numCols)
    {
        d_mat[IDX_R(headRowIdx, colIdx, numRows, numCols)] = dataHeads[colIdx] / sqrt((double)numRows);
        // printf("HT: %.3f\n", dataHeads[colIdx] / sqrt(numRows));
    }
}

// computeHeadsAndTails<<<dOffsets2.size(), numCols2>>>(
//     matCuda2.getData(),
//     dOffsets2,
//     numCols2);

template<typename T>
__global__ void computeHeadsAndTails(T* d_mat, const int* d_offsets, int numCols) {
    extern __shared__ T dataHeads  [];
    int colIdx = threadIdx.x;
    int offsetIdx = blockIdx.x;
    int headRowIdx = d_offsets[offsetIdx];
    int numRows = d_offsets[offsetIdx + 1] - d_offsets[offsetIdx];

    if (colIdx < numCols)
    {
        dataHeads[colIdx] = d_mat[IDX_R(headRowIdx, colIdx, numRows, numCols)];
    }
    __syncthreads();
    for (int rowIdx = headRowIdx + 1; rowIdx < numRows; rowIdx++)
    {
        T i = rowIdx - headRowIdx + 1;
        if (colIdx < numCols)
        {
            T prevRowSum;
            T tailVal;
            prevRowSum = dataHeads[colIdx];
            T matVal = d_mat[IDX_R(rowIdx, colIdx, numRows, numCols)];
            dataHeads[colIdx] += matVal;
            tailVal = (matVal * (i - 1) - prevRowSum) / sqrt(i * (i - 1));
            d_mat[IDX_R(rowIdx, colIdx, numRows, numCols)] = tailVal;
            // printf("TAIL VAL %d %d %.3f %.3f\n", rowIdx, colIdx, i, tailVal);
        }
        __syncthreads();
    }
    if (colIdx < numCols)
    {
        d_mat[IDX_R(headRowIdx, colIdx, numRows, numCols)] = dataHeads[colIdx] / sqrt((double)numRows);
        // printf("HT: %.3f\n", dataHeads[colIdx] / sqrt(numRows));
    }
}

template <typename T>
__global__ void concatenateHeadsAndTails(const T* d_mat, const T* d_mat2Mod, T* dOutMat, int numRows1, int numCols1, int numRows2, int numCols2) {
    int colIdx = threadIdx.x;
    int headRowIdx = 0;
    const int numRowsOut = numRows1 + numRows2 - 1;
    const int numColsOut = numCols1 + numCols2;

    for (int rowIdx = headRowIdx; rowIdx < headRowIdx + numRows1; rowIdx++)
    {
        int outRowIdx = rowIdx;
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(outRowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = d_mat[IDX_R(rowIdx, colIdx, numRows1, numCols1)] * sqrt((double)numRows2);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = IDX_R(outRowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[IDX_R(headRowIdx, colIdx, numRows2, numCols2)];
        }
    }
    for (int rowIdx = headRowIdx + numRows1; rowIdx < headRowIdx + numRowsOut; rowIdx++)
    {
        int outRowIdx = rowIdx;
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(outRowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = 0;
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = IDX_R(outRowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[IDX_R(rowIdx - numRows1 + 1, colIdx, numRows2, numCols2)] * sqrt((double)numRows1);
        }
    }
}

template <typename T>
__global__ void setZerosUpperTriangular(T* d_A, int numRows, int numCols)
{
	int colIdx = threadIdx.x;
	for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
	{
		if (rowIdx > colIdx)
		{
			d_A[IDX_C(rowIdx, colIdx, numRows, numCols)] = 0;
		}
	}
}

// TODO: Add inverse, matrix matrix multiplication

template <typename T>
__global__ void findUniqueOffsets(const T* d_arr, int* d_offsets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || (idx < n && d_arr[idx] != d_arr[idx - 1]))
    {
        d_offsets[idx] = idx;
    }
    else if (idx < n)
    {
        d_offsets[idx] = -1;
    }
}

template <typename T>
void computeOffsets(const MatrixCudaRow<T>& matA, thrust::device_vector<int>& dOut)
{
    int numRows = matA.getNumRows();
    thrust::device_vector<int> d_offsets(numRows);

    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;
    auto matAJoinCol = matA.getColumn(0);
    // std::cout << matAJoinCol << std::endl;
    findUniqueOffsets<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(matAJoinCol.getDataC()), thrust::raw_pointer_cast(d_offsets.data()), numRows);

    int countDif = thrust::count(d_offsets.begin(), d_offsets.end(), -1);
    dOut = std::move(thrust::device_vector<int>(countDif + 1));
    thrust::copy_if(d_offsets.begin(), d_offsets.end(), dOut.begin(), [] __device__ (int val) { return val != -1; });
    dOut.push_back(matA.getNumRows());
}

template <typename T>
int computeFigaro(const MatrixRow<T>& mat1, const MatrixRow<T>& mat2,
    MatrixCol<T>& matR, MatrixCol<T>& matQ, const std::string& fileName, ComputeDecomp decompType)
{
    int numRows1 = mat1.getNumRows();
    int numCols1 = mat1.getNumCols();
    int numRows2 = mat2.getNumRows();
    int numCols2 = mat2.getNumCols();
    int numRowsOut = numRows1 + numRows2 - 1;
    int numColsOut = numCols1 + numCols2;

    MatrixCudaRow<T> matCuda1(mat1);
    MatrixCudaRow<T> matCuda2(mat2);
    MatrixCudaRow<T> matCudaOut(numRowsOut, numColsOut);
    MatrixCudaCol<T> matCudaTran(numRowsOut, numColsOut);
    thrust::device_vector<int> dOffsets1;
    thrust::device_vector<int> dOffsets2;

    // std::cout << "C1" << matCuda1;
    computeOffsets(matCuda1, dOffsets1);
    // std::cout << "C2" << matCuda2;
    computeOffsets(matCuda2, dOffsets2);
    // TODO: Compute sizes of chunks
    // TODO: computeHeadsAndTails for the array of entries
    // TODO: concatenateHeadsAndTails for the array of entries.


    T *d_S;
    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY;
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));

    // Compute buffer size for QR
    int workspace_size = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf_bufferSize(cusolverH, numRowsOut, numColsOut, matCudaOut.getData(), numRowsOut, &workspace_size));
    }
    else
    {
        CUSOLVER_CALL(cusolverDnDgeqrf_bufferSize(cusolverH, numRowsOut, numColsOut, matCudaOut.getData(), numRowsOut, &workspace_size));
    }

    // Initialize cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate workspace
    T *d_work, *d_tau;
    CUDA_CALL(cudaMalloc((void**)&d_work, workspace_size * sizeof(T)));

    // Allocate device status variable
    int *devInfo;
    CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_tau, std::min(numRowsOut, numColsOut) * sizeof(T)));

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // Start measuring time
    CUDA_CALL(cudaEventRecord(start));

    // Compute join offsets for both tables
    // compute join offsets
    // for loop call for each subset the
    // We assume there are no dangling tuples.
    // std::host_vector<T*> vMatCuda2Ptrs(dOffsets2.size(), nullptr);
    // std::host_vector<int> vNumRows1(dOffsets1.size(), nullptr);
    // std::host_vector<int > vNumRows2(dOffsets2.size(), nullptr);
    // computeHeadsAndTails<<<dOffsets2.size(), numCols2, numCols2>>>(
    //     matCuda2.getData(),
    //     dOffsets2.data().get(),
    //     numCols2);

    computeHeadsAndTails<<<1, numCols2, numCols2>>>(matCuda2.getData(), numRows2, numCols2);
    // concatenateHeadsAndTails<<<dOffsets2.size(), max(numCols1, numCols2)>>>(
        //matCuda1.getDataC(),
        // matCuda2.getDataC(),
        // matCudaOut.getData(),
        // dOffsets1.data().get(),
        // numCols1,
        // dOffsets2.data().get(),
        //numCols2,
        // );
    concatenateHeadsAndTails<<<1, max(numCols1, numCols2)>>>(matCuda1.getDataC(), matCuda2.getDataC(), matCudaOut.getData(), numRows1, numCols1, numRows2, numCols2);

    // Define scalars alpha and beta
    const T alpha = 1.0f; // Scalar for matrix A (no scaling)
    const T beta = 0.0f;  // Scalar for matrix B (no B matrix, so no scaling)

    if constexpr (std::is_same<T, float>::value)
    {
        cublasSgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
        numRowsOut, numColsOut,                     // Matrix dimensions
        &alpha,                   // Scalar for A
        matCudaOut.getDataC(), numColsOut,                 // Input matrix A and its leading dimension
        &beta,                    // Scalar for B (not used)
        nullptr, numColsOut,               // No B matrix (set to nullptr)
        matCudaTran.getData(), numRowsOut);                  // Output matrix C and its leading dimension
    }
    else
    {
        cublasDgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
        numRowsOut, numColsOut,                     // Matrix dimensions
        &alpha,                   // Scalar for A
        matCudaOut.getDataC(), numColsOut,                   // Input matrix A and its leading dimension
        &beta,                    // Scalar for B (not used)
        nullptr, numColsOut,               // No B matrix (set to nullptr)
        matCudaTran.getData(), numRowsOut);                  // Output matrix C and its leading dimension
    }

    int rank = min(numRowsOut, numColsOut);

    // Compute QR factorization
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf(cusolverH, numRowsOut, numColsOut, matCudaTran.getData(), numRowsOut, d_tau, d_work, workspace_size, devInfo));
    }
    else
    {
        CUSOLVER_CALL(cusolverDnDgeqrf (cusolverH, numRowsOut, numColsOut, matCudaTran.getData(), numRowsOut, d_tau, d_work, workspace_size, devInfo));
        setZerosUpperTriangular<<<1, numColsOut>>>(matCudaTran.getData(), numRowsOut, numColsOut);
    	if (computeSVD)
	    {
            std::cout << "WTF" << std::endl;
            char jobu = 'N';  // No computation of U
            char jobvt = 'N'; // No computation of V^T
            // cuSOLVER handle
            int *d_info;
            double *d_work;
            int lwork = 0;
            int ldA = numRowsOut;

            cusolverDnHandle_t cusolverH1 = nullptr;
            CUSOLVER_CALL(cusolverDnCreate(&cusolverH1));
            CUDA_CALL(cudaMalloc((void**)&d_info, sizeof(int)));
            CUSOLVER_CALL(cusolverDnDgesvd_bufferSize(cusolverH, rank, numColsOut, &lwork));
            CUDA_CALL(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
                CUDA_CALL(cudaMalloc((void**)&d_S, sizeof(double) * rank));
            cusolverDnDgesvd(cusolverH1, jobu, jobvt, numColsOut, numColsOut, matCudaTran.getData(), ldA, d_S, nullptr, numColsOut, nullptr, numColsOut,
                            d_work, lwork, nullptr, d_info);
        }
    }

    // Stop measuring time
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    if (computeSVD)
    {
        thrust::host_vector<T> h_matS(numColsOut);
        T *h_S = thrust::raw_pointer_cast(h_matS.data());
        CUDA_CALL(cudaMemcpy(h_S, d_S, numColsOut * sizeof(T), cudaMemcpyDeviceToHost));
        printMatrix<T, MajorOrder::COL_MAJOR>(h_S, numColsOut, 1, numColsOut, fileName + "LinScaleS", false);
    }
    else
    {
    	thrust::host_vector<T> h_matOutH(numRowsOut * numColsOut);
    	T *h_matOut = thrust::raw_pointer_cast(h_matOutH.data());
    	CUDA_CALL(cudaMemcpy(h_matOut, matCudaTran.getDataC(), numRowsOut * numColsOut * sizeof(T), cudaMemcpyDeviceToHost));
        matR = Matrix<T, MajorOrder::COL_MAJOR>{numColsOut, numColsOut};
        copyMatrix<T, MajorOrder::COL_MAJOR>(h_matOut, matR.getData(), numRowsOut, numColsOut, numColsOut, numColsOut, false);
    }


    CUDA_CALL(cudaFree(d_tau));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(devInfo));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));

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
int computeGeneral(const Matrix<T, majorOrder>& matA, MatrixCol<T>& matR, const std::string& fileName, ComputeDecomp decompType)
{
    T *d_tau, *h_S;
    int numRows = matA.getNumRows();
    int numCols = matA.getNumCols();

    MatrixCudaCol<T> matACuda(matA);
    MatrixCol<T> matOut(numRows, numCols);
    MatrixCudaCol<T> matACudaCol(numRows, numCols);
    thrust::host_vector<T> h_matS(numCols);

    h_S = thrust::raw_pointer_cast(h_matS.data());
    T *d_S;
    CUDA_CALL(cudaMalloc((void**)&d_tau, std::min(numRows, numCols) * sizeof(T)));
    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY;

    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        const T alpha = 1.0f;
        const T beta = 0.0f;

        if constexpr (std::is_same<T, float>::value)
        {
            cublasSgeam(handle,
            CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
            numRows, numCols,                     // Matrix dimensions
            &alpha,                   // Scalar for A
            matACuda.getDataC(), numCols,                   // Input matrix A and its leading dimension
            &beta,                    // Scalar for B (not used)
            nullptr, numCols,               // No B matrix (set to nullptr)
            matACudaCol.getData(), numRows);                  // Output matrix C and its leading dimension
        }
        else
        {
            cublasDgeam(handle,
            CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
            numRows, numCols,                     // Matrix dimensions
            &alpha,                   // Scalar for A
            matACuda.getDataC(), numCols,                   // Input matrix A and its leading dimension
            &beta,                    // Scalar for B (not used)
            nullptr, numCols,               // No B matrix (set to nullptr)
            matACudaCol.getData(), numRows);                  // Output matrix C and its leading dimension
        }
        cublasDestroy(handle);
    }
    else
    {
        matACudaCol = std::move(matACuda);
    }

    cusolverDnHandle_t cusolverH;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));

    int workspace_size = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf_bufferSize(cusolverH, numRows, numCols, matACudaCol.getData(), numRows, &workspace_size));
    }
    else
    {
        CUSOLVER_CALL(cusolverDnDgeqrf_bufferSize(cusolverH, numRows, numCols, matACudaCol.getData(), numRows, &workspace_size));
    }
    T *d_work;
    int *devInfo;
    CUDA_CALL(cudaMalloc((void**)&d_work, workspace_size * sizeof(T)));
    CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)));

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf(cusolverH, numRows, numCols, matACudaCol.getData(), numRows, d_tau, d_work, workspace_size, devInfo));
    }
    else
    {
        if (computeSVD)
        {
            char jobu = 'N';  // No computation of U
            char jobvt = 'N'; // No computation of V^T

            int *d_info;
            double *d_work;
            int lwork = 0;
            int ldA = numRows;

            cusolverDnHandle_t cusolverH1 = nullptr;
            CUSOLVER_CALL(cusolverDnCreate(&cusolverH1));
            CUDA_CALL(cudaMalloc((void**)&d_info, sizeof(int)));
            CUSOLVER_CALL(cusolverDnDgesvd_bufferSize(cusolverH, numRows, numCols, &lwork));
            CUDA_CALL(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
            CUDA_CALL(cudaMalloc((void**)&d_S, sizeof(double) * numCols));

            cusolverDnDgesvd(cusolverH1, jobu, jobvt, numRows, numCols, matACudaCol.getData(), ldA, d_S, nullptr, numRows, nullptr, numCols,
                                    d_work, lwork, nullptr, d_info);
        }
        else
        {
            CUSOLVER_CALL(cusolverDnDgeqrf(cusolverH, numRows, numCols, matACudaCol.getData(), numRows, d_tau, d_work, workspace_size, devInfo));
        }
    }

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results back to host
    if (computeSVD)
    {
        CUDA_CALL(cudaMemcpy(h_S, d_S, numCols * sizeof(T), cudaMemcpyDeviceToHost));
        printMatrix<T, MajorOrder::COL_MAJOR>(h_S, numCols, 1, numCols, fileName + "cuSolverS", false);
    }
    else
    {
        CUDA_CALL(cudaMemcpy(matOut.getData(), matACudaCol.getDataC(), matACudaCol.getSize(), cudaMemcpyDeviceToHost));
        matR = Matrix<T, MajorOrder::COL_MAJOR>{numCols, numCols};
        copyMatrix<T, MajorOrder::COL_MAJOR>(matOut.getDataC(), matR.getData(), numRows, numCols, numCols, numCols, true);
    }

    // Print execution time
    std::string nameDecomp = computeSVD ? "SVD" : "QR";
    std::cout << "\n" + nameDecomp + " decomposition CUSolver took " << milliseconds << " ms.\n";

    CUDA_CALL(cudaFree(d_tau));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(devInfo));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));

    return 0;
}

// template int computeGeneral<double, MajorOrder::ROW_MAJOR>(const MatrixDRow& matA,
//     MatrixDCol& matR, const std::string& fileName, ComputeDecomp decompType);

template int computeGeneral<double, MajorOrder::COL_MAJOR>(const MatrixDCol& matA,
        MatrixDCol& matR, const std::string& fileName, ComputeDecomp decompType);

template int computeFigaro<double>(const MatrixDRow& mat1, const MatrixDRow& mat2,
    Matrix<double, MajorOrder::COL_MAJOR>& matR, Matrix<double, MajorOrder::COL_MAJOR>& matQ, const std::string& fileName, ComputeDecomp decompType);