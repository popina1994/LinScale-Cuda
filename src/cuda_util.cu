#include "cuda_util.h"
std::map<std::string, double> mapMemoryTrack;
std::map<std::string, double> mapMemoryStart;

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch (status) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "Library not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "Resource allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "Invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "Architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "Memory mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "Execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "Internal error";
        default:
            return "Unknown error";
    }
}

const char* cusolverGetErrorString(cusolverStatus_t status)
{
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
        case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
        default: return "Unknown cuSOLVER error";
    }
}

double getCudaMemoryUsage(void)
{
    size_t freeBytes;
    size_t totalBytes;

    cudaError_t status = cudaMemGetInfo(&freeBytes, &totalBytes);

    auto freeMB = freeBytes / (1024.0 * 1024);
    auto totalMB = totalBytes / (1024.0 * 1024);

    return totalMB - freeMB;
}
