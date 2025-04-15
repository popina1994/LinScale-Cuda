cmake_minimum_required(VERSION 3.22)
project(LinScale-Cuda LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)

find_package(BLAS REQUIRED)
find_package(MKL REQUIRED)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "-O3")
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda")

file(GLOB_RECURSE CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cxx
)
file(GLOB_RECURSE CUDA_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cu
)
if(DEFINED MEMORY_USAGE)
    message("")
else()
    set(MEMORY_USAGE "1")
endif()

add_executable(${PROJECT_NAME} ${CPP_SOURCES})

add_library(cuda_lib STATIC
    ${CUDA_SOURCES}
)
target_compile_definitions(cuda_lib PRIVATE MEMORY_USAGE=${MEMORY_USAGE})
target_link_libraries(cuda_lib PRIVATE cudart cublas cusolver)

target_include_directories(cuda_lib PRIVATE ${CUDA_INCLUDE_DIRS} header)

target_link_libraries(${PROJECT_NAME} PRIVATE cuda_lib)
target_include_directories(${PROJECT_NAME} PRIVATE ${MKL_INCLUDE_DIRS} header)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system thread iostreams program_options)
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL_LIBRARIES} ${BLAS_LIBRARIES}
    ${Boost_LIBRARIES}
)

# Optional: Use static MKL linking
# set(MKL_THREADING "sequential")  # or "tbb" or "openmp"
# set(MKL_INTERFACE "lp64")       # or "ilp64"
