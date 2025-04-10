#ifndef _LINSCALE_CUDA_H_
#define _LINSCALE_CUDA_H_


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


#endif