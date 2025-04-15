#ifndef _LINSCALE_CUDA_H_
#define _LINSCALE_CUDA_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <string>
#include <map>

#ifndef MEMORY_USAGE
#define MEMORY_USAGE 1
#endif

const char* cublasGetErrorString(cublasStatus_t status);
const char* cusolverGetErrorString(cusolverStatus_t status);
double getCudaMemoryUsage(void);

// CUDA error check macro
#define CUDA_CALL(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    } \
} while (0)




#define CUBLASS_CALL(call) \
{ \
    auto status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << cublasGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);\
    }\
}



// cuSOLVER error check macro
#define CUSOLVER_CALL(call) \
do { \
    cusolverStatus_t err = call; \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER Error: " << cusolverGetErrorString(err) << " at " << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    } \
} while (0)

extern std::map<std::string, double> mapMemoryTrack;
extern std::map<std::string, double> mapMemoryStart;

#if MEMORY_USAGE == 1
#define MEMORY_LOG(TAG, MESSAGE)         \
do {                                     \
    auto memUsed = getCudaMemoryUsage(); \
    double maxMem = 0.0;                 \
    if (mapMemoryTrack.find(TAG) != mapMemoryTrack.end())    \
    {                                    \
        maxMem = mapMemoryTrack[TAG]; \
    }                                    \
    else \
    { \
        mapMemoryStart[TAG] = memUsed;    \
    }\
    maxMem = std::max(maxMem, memUsed - mapMemoryStart[TAG]);  \
    mapMemoryTrack[TAG] = maxMem;    \
    std::cout << MESSAGE << " " << memUsed << "MB" << std::endl; \
} while (0);
#else
#define MEMORY_LOG(TAG, MESSAGE)
#endif

#endif