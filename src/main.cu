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
#include "types.h"

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

#define IDX(rowIdx, colIdx, width) ((rowIdx) * (width) + (colIdx))
#define IDX_R(rowIdx, colIdx, numRows, numCols) ((rowIdx) * (numCols) + (colIdx) )
#define IDX_C(rowIdx, colIdx, numRows, numCols) ((rowIdx)  + (colIdx) * (numRows))

template <typename T, MajorOrder order>
void printMatrix(T* pArr, int numRows, int numCols, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?" << fileName << std::endl;
    }
    outFile << std::fixed << std::setprecision(15);
    for (int rowIdx = 0; rowIdx < min(numRows, numRowsCut); rowIdx++)
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
__global__ void concatenateHeadsAndTails(T* d_mat, T* d_mat2Mod, T* dOutMat, int numRows1, int numCols1, int numRows2, int numCols2) {
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

template <typename T>
int computeFigaro(T* h_mat1, T* h_mat2, int numRows1, int numCols1, int numRows2, int numCols2,
    std::string& fileName, int compute)
{
    int numRowsOut = numRows1 + numRows2 - 1;
    int numColsOut = numCols1 + numCols2;

    thrust::device_vector<T> d_mat1DV(h_mat1, h_mat1 + numRows1 * numCols1);
    thrust::device_vector<T> d_mat2DV(h_mat2, h_mat2 + numRows2 * numCols2);
    thrust::device_vector<T> d_matOutDV(numRowsOut * numColsOut);
    thrust::device_vector<T> d_matTranDV(numRowsOut * numColsOut);
    thrust::host_vector<T> h_tmpOutOrig(numRowsOut * numColsOut);
    thrust::host_vector<T> h_tmpOutTran(numRowsOut * numColsOut);

    T* d_mat1 = thrust::raw_pointer_cast(d_mat1DV.data());
    T *d_mat2 = thrust::raw_pointer_cast(d_mat2DV.data());
    T* d_matOut = thrust::raw_pointer_cast(d_matOutDV.data());
    T *d_S;
    T* d_matOutTran = thrust::raw_pointer_cast(d_matTranDV.data());
    T* h_pTmpOutOrig =  thrust::raw_pointer_cast(h_tmpOutOrig.data());
    T* h_pTmpOutTran =  thrust::raw_pointer_cast(h_tmpOutTran.data());
    bool computeSVD = compute == 2;
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CALL(cusolverDnCreate(&cusolverH));

    // Compute buffer size for QR
    int workspace_size = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf_bufferSize(cusolverH, numRowsOut, numColsOut, d_matOut, numRowsOut, &workspace_size));
    }
    else
    {
        CUSOLVER_CALL(cusolverDnDgeqrf_bufferSize(cusolverH, numRowsOut, numColsOut, d_matOut, numRowsOut, &workspace_size));
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
    computeHeadsAndTails<<<1, numCols2>>>(d_mat2, numRows2, numCols2);
    concatenateHeadsAndTails<<<1, max(numCols1, numCols2)>>>(d_mat1, d_mat2, d_matOut, numRows1, numCols1, numRows2, numCols2);

    // Define scalars alpha and beta
    const T alpha = 1.0f; // Scalar for matrix A (no scaling)
    const T beta = 0.0f;  // Scalar for matrix B (no B matrix, so no scaling)

    CUDA_CALL(cudaMemcpy(h_pTmpOutOrig, d_matOut, numRowsOut * numColsOut * sizeof(T), cudaMemcpyDeviceToHost));
    if constexpr (std::is_same<T, float>::value)
    {
        cublasSgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
        numRowsOut, numColsOut,                     // Matrix dimensions
        &alpha,                   // Scalar for A
        d_matOut, numColsOut,                   // Input matrix A and its leading dimension
        &beta,                    // Scalar for B (not used)
        nullptr, numColsOut,               // No B matrix (set to nullptr)
        d_matOutTran, numRowsOut);                  // Output matrix C and its leading dimension
    }
    else
    {
        cublasDgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_T, // Transpose A (CUBLAS_OP_T), no transpose for B (CUBLAS_OP_N)
        numRowsOut, numColsOut,                     // Matrix dimensions
        &alpha,                   // Scalar for A
        d_matOut, numColsOut,                   // Input matrix A and its leading dimension
        &beta,                    // Scalar for B (not used)
        nullptr, numColsOut,               // No B matrix (set to nullptr)
        d_matOutTran, numRowsOut);                  // Output matrix C and its leading dimension
    }

    CUDA_CALL(cudaMemcpy(h_pTmpOutTran, d_matOutTran, numRowsOut * numColsOut * sizeof(T), cudaMemcpyDeviceToHost));
    int rank = min(numRowsOut, numColsOut);

    // // Compute QR factorization
    if constexpr (std::is_same<T, float>::value)
    {
        CUSOLVER_CALL(cusolverDnSgeqrf(cusolverH, numRowsOut, numColsOut, d_matOutTran, numRowsOut, d_tau, d_work, workspace_size, devInfo));
    }
    else
    {
        CUSOLVER_CALL(cusolverDnDgeqrf (cusolverH, numRowsOut, numColsOut, d_matOutTran, numRowsOut, d_tau, d_work, workspace_size, devInfo));
    	if (computeSVD)
	    {
            std::cout << "WTF" << std::endl;
            setZerosUpperTriangular<<<1, numColsOut>>>(d_matOutTran, numRowsOut, numColsOut);
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
            cusolverDnDgesvd(cusolverH1, jobu, jobvt, numColsOut, numColsOut, d_matOutTran, ldA, d_S, nullptr, numColsOut, nullptr, numColsOut,
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
    	CUDA_CALL(cudaMemcpy(h_matOut, d_matOutTran, numRowsOut * numColsOut * sizeof(T), cudaMemcpyDeviceToHost));
        printMatrix<T, MajorOrder::COL_MAJOR>(h_matOut, numRowsOut, numColsOut, numColsOut, fileName + "LinScale", true);

        printMatrix<T, MajorOrder::ROW_MAJOR>(h_pTmpOutOrig, numRowsOut, numColsOut, numColsOut, fileName + "LinScaleOrig", false);
        printMatrix<T, MajorOrder::COL_MAJOR>(h_pTmpOutTran, numRowsOut, numColsOut, numColsOut, fileName + "LinScaleTran", false);
    }


    CUDA_CALL(cudaFree(d_tau));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(devInfo));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    CUSOLVER_CALL(cusolverDnDestroy(cusolverH));

    // delete [] h_mat1;
    // delete [] h_mat2;
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
extern "C++" {
template <typename T, MajorOrder majorOrder>
int computeGeneral(T* h_A, int numRows, int numCols, const std::string& fileName, int compute)
{
    // Allocate device memory
    T *d_A, *d_tau, *d_matOutTran, *h_S;

    thrust::device_vector<T> d_matA(h_A, h_A + numRows * numCols);
    thrust::device_vector<T> d_matADV(numRows * numCols);
    thrust::host_vector<T> h_matS(numCols);

    d_A = thrust::raw_pointer_cast(d_matA.data());
    d_matOutTran = thrust::raw_pointer_cast(d_matADV.data());
    h_S = thrust::raw_pointer_cast(h_matS.data());
    T *d_S;
    CUDA_CALL(cudaMalloc((void**)&d_tau, std::min(numRows, numCols) * sizeof(T)));
    bool computeSVD = compute == 2;
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
        CUDA_CALL(cudaMemcpy(h_A, d_matOutTran, numRows * numCols * sizeof(T), cudaMemcpyDeviceToHost));
    }

    CUDA_CALL(cudaMemcpy(h_A, d_matOutTran, numRows * numCols * sizeof(T), cudaMemcpyDeviceToHost));

    printMatrix<T, MajorOrder::COL_MAJOR>(h_A, numRows, numCols, numCols, fileName + "CUDA", true);

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
}

template int computeGeneral<double, MajorOrder::ROW_MAJOR>(double* h_A, int numRows, int numCols, const std::string& fileName, int compute);

template int computeGeneral<double, MajorOrder::COL_MAJOR>(double* h_A, int numRows, int numCols, const std::string& fileName, int compute);

template int computeFigaro<double>(double* h_mat1, double* h_mat2, int numRows1, int numCols1, int numRows2, int numCols2,
    std::string& fileName, int compute);