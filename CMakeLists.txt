cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 23)

project(rank1_covariance CUDA)

add_executable(rank1_covariance kernels.hpp kernels.cu main.cu )

set_target_properties(rank1_covariance PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(rank1_covariance PRIVATE ${CUDA_LIBRARIES} cublas)