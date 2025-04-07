#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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
    __shared__ T dataHeads  [1024];
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

template <typename T>
__global__ void concatenateHeadsAndTails(const T* d_mat, const T* d_mat2Mod, T* dOutMat, int numRows1, int numCols1, int numRows2, int numCols2) {
    int colIdx = threadIdx.x;
    int headRowIdx = 0;
    const int numRowsOut = numRows1 + numRows2 - 1;
    const int numColsOut = numCols1 + numCols2;

    for (int rowIdx = 0; rowIdx < numRows1; rowIdx++)
    {
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(rowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = d_mat[IDX_R(rowIdx, colIdx, numRows1, numCols1)] * sqrt((double)numRows2);
            // printf("HERE 1 %d %d %.3f %d\n", rowIdx, colIdx, dOutMat[posIdx], posIdx);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = IDX_R(rowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[IDX_R(headRowIdx, colIdx, numRows2, numCols2)];
            // printf("HERE 1 %d %d %.3f %d\n", rowIdx, colIdx + numCols1, dOutMat[posIdx2], posIdx2);
        }
    }
    for (int rowIdx = numRows1; rowIdx < numRowsOut; rowIdx++)
    {
        if (colIdx < numCols1)
        {
            int posIdx = IDX_R(rowIdx, colIdx, numRowsOut, numColsOut);
            dOutMat[posIdx] = 0;
            // printf("HERE 2 %d %d %.3f %d \n", rowIdx, colIdx, dOutMat[posIdx], posIdx);
        }
        if (colIdx < numCols2)
        {
            int posIdx2 = IDX_R(rowIdx, colIdx + numCols1, numRowsOut, numColsOut);
            dOutMat[posIdx2] = d_mat2Mod[IDX_R(rowIdx - numRows1 + 1, colIdx, numRows2, numCols2)] * sqrt((double)numRows1);
            // printf("HERE 2 %d %d %.3f %d\n", rowIdx, colIdx + numCols1, dOutMat[posIdx2], posIdx2);
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

__global__ void findUniqueOffsets(const int* d_arr, int* d_offsets, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || (idx < n && d_arr[idx] != d_arr[idx - 1])) {
        d_offsets[idx] = idx;
    } else if (idx < n) {
        d_offsets[idx] = -1;
    }
}

// void computeIndices(thrust::device_vector<int>& d_indices, thrust::device_vector<int>& d_indices)
// {
//     const int n = 12;
//     int h_arr[n] = {1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 7};
//     int h_offsets[n];

//     // Allocate memory on device
//     thrust::device_vector<int> d_arr(h_arr, h_arr + n);
//     thrust::device_vector<int> d_offsets(n);

//     int blockSize = 256;
//     int numBlocks = (n + blockSize - 1) / blockSize;
//     findUniqueOffsets<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_arr.data()), thrust::raw_pointer_cast(d_offsets.data()), n);

//     // Step to remove -1 entries (non-starting positions)
//     int countDif = thrust::count(d_offsets, begin(), d_offsets.end(), -1);
//     thrust::device_vector<int> d_result(countDif);
//     auto new_end = thrust::copy_if(d_offsets.begin(), d_offsets.end(), d_result.begin(), [] __device__ (int val) { return val != -1; });

//     // Sort the offsets (in case atomic ordering caused non-determinism)
//     thrust::sort(d_result.begin(), d_result.end());

//     // Copy result to host and print
//     thrust::host_vector<int> h_result = d_result;
// }

template <typename T>
int computeFigaro(const MatrixDRow& mat1, const MatrixDRow& mat2,
    Matrix<T, MajorOrder::COL_MAJOR>& matR, Matrix<T, MajorOrder::COL_MAJOR>& matQ, const std::string& fileName, ComputeDecomp decompType)
{
    int numRows1 = mat1.getNumRows();
    int numCols1 = mat1.getNumCols();
    int numRows2 = mat2.getNumRows();
    int numCols2 = mat2.getNumCols();
    int numRowsOut = numRows1 + numRows2 - 1;
    int numColsOut = numCols1 + numCols2;

    MatrixCuda<T, MajorOrder::ROW_MAJOR> matCuda1(mat1);
    MatrixCuda<T, MajorOrder::ROW_MAJOR> matCuda2(mat2);
    MatrixCuda<T, MajorOrder::ROW_MAJOR> matCudaOut(numRowsOut, numColsOut);
    MatrixCuda<T, MajorOrder::COL_MAJOR> matCudaTran(numRowsOut, numColsOut);

    T *d_S;
    T* d_matOutTran = matCudaTran.getData();
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
    computeHeadsAndTails<<<1, numCols2>>>(matCuda2.getData(), numRows2, numCols2);
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
        matCudaOut.getDataC(), numColsOut,                   // Input matrix A and its leading dimension
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
int computeGeneral(const Matrix<T, majorOrder>& matA, Matrix<T, MajorOrder::COL_MAJOR>& matR, const std::string& fileName, ComputeDecomp decompType)
{
    // Allocate device memory
    T *d_A, *d_tau, *d_matOutTran, *h_S, *h_aCopy;
    int numRows = matA.getNumRows();
    int numCols = matA.getNumCols();

    thrust::device_vector<T> d_matA(matA.getDataC(), matA.getDataC() + numRows * numCols);
    thrust::host_vector<T> h_matACopy(numRows * numCols);
    thrust::device_vector<T> d_matADV(numRows * numCols);
    thrust::host_vector<T> h_matS(numCols);

    d_A = thrust::raw_pointer_cast(d_matA.data());
    d_matOutTran = thrust::raw_pointer_cast(d_matADV.data());
    h_S = thrust::raw_pointer_cast(h_matS.data());
    h_aCopy = thrust::raw_pointer_cast(h_matACopy.data());
    T *d_S;
    CUDA_CALL(cudaMalloc((void**)&d_tau, std::min(numRows, numCols) * sizeof(T)));
    bool computeSVD = decompType == ComputeDecomp::SIGMA_ONLY;
     // Copy data to GPU
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        // Initialize cuBLAS handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Define scalars alpha and beta
        const T alpha = 1.0f; // Scalar for matrix A (no scaling)
        const T beta = 0.0f;  // Scalar for matrix B (no B matrix, so no scaling)

        if constexpr (std::is_same<T, float>::value)
        {
            cublasSgeam(handle,
            CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
            numRows, numCols,                     // Matrix dimensions
            &alpha,                   // Scalar for A
            d_A, numCols,                   // Input matrix A and its leading dimension
            &beta,                    // Scalar for B (not used)
            nullptr, numCols,               // No B matrix (set to nullptr)
            d_matOutTran, numRows);                  // Output matrix C and its leading dimension
        }
        else
        {
            cublasDgeam(handle,
            CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
            numRows, numCols,                     // Matrix dimensions
            &alpha,                   // Scalar for A
            d_A, numCols,                   // Input matrix A and its leading dimension
            &beta,                    // Scalar for B (not used)
            nullptr, numCols,               // No B matrix (set to nullptr)
            d_matOutTran, numRows);                  // Output matrix C and its leading dimension
        }
        cublasDestroy(handle);
    }
    else
    {
        d_matOutTran = d_A;
    }

    // cuSOLVER handle
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));

    // Compute buffer size for QR
    int workspace_size = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf_bufferSize(cusolverH, numRows, numCols, d_matOutTran, numRows, &workspace_size));
    }
    else
    {
        CUSOLVER_CALL(cusolverDnDgeqrf_bufferSize(cusolverH, numRows, numCols, d_matOutTran, numRows, &workspace_size));
    }
    // Allocate workspace
    T *d_work;
    CUDA_CALL(cudaMalloc((void**)&d_work, workspace_size * sizeof(T)));

    // Allocate device status variable
    int *devInfo;
    CUDA_CALL(cudaMalloc((void**)&devInfo, sizeof(int)));

    // CUDA event timing variables
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // Start measuring time
    CUDA_CALL(cudaEventRecord(start));

    // Compute QR factorization
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf(cusolverH, numRows, numCols, d_matOutTran, numRows, d_tau, d_work, workspace_size, devInfo));
    }
    else
    {
        if (computeSVD)
        {
            char jobu = 'N';  // No computation of U
            char jobvt = 'N'; // No computation of V^T
            // cuSOLVER handle
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

            cusolverDnDgesvd(cusolverH1, jobu, jobvt, numRows, numCols, d_matOutTran, ldA, d_S, nullptr, numRows, nullptr, numCols,
                                    d_work, lwork, nullptr, d_info);
        }
        else
        {
            CUSOLVER_CALL(cusolverDnDgeqrf(cusolverH, numRows, numCols, d_matOutTran, numRows, d_tau, d_work, workspace_size, devInfo));
        }
    }

    // Stop measuring time
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    // Compute elapsed time
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
        CUDA_CALL(cudaMemcpy(h_aCopy, d_matOutTran, numRows * numCols * sizeof(T), cudaMemcpyDeviceToHost));
        matR = Matrix<T, MajorOrder::COL_MAJOR>{numCols, numCols};
        copyMatrix<T, MajorOrder::COL_MAJOR>(h_aCopy, matR.getData(), numRows, numCols, numCols, numCols, true);
    }

    // Print execution time
    std::string nameDecomp = computeSVD ? "SVD" : "QR";
    std::cout << "\n" + nameDecomp + " decomposition CUSolver took " << milliseconds << " ms.\n";

    // Cleanup
    CUDA_CALL(cudaFree(d_tau));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(devInfo));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));

    return 0;
}

template int computeGeneral<double, MajorOrder::ROW_MAJOR>(const MatrixDRow& matA,
    MatrixDCol& matR, const std::string& fileName, ComputeDecomp decompType);

template int computeGeneral<double, MajorOrder::COL_MAJOR>(const MatrixDCol& matA,
        MatrixDCol& matR, const std::string& fileName, ComputeDecomp decompType);

template int computeFigaro<double>(const MatrixDRow& mat1, const MatrixDRow& mat2,
    Matrix<double, MajorOrder::COL_MAJOR>& matR, Matrix<double, MajorOrder::COL_MAJOR>& matQ, const std::string& fileName, ComputeDecomp decompType);