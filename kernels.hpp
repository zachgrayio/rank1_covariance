#ifndef KERNELS_H
#define KERNELS_H

namespace kernels {
    __global__ void rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n);
    __global__ void shared_rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n);
    __global__ void update_means_inplace(const float* d_data, float* d_mean, const int rows, const int cols);
    __global__ void subtract_means(const float* d_data, const float* d_mean, float* d_centered, const int rows, const int cols);
    __global__ void update_covariance_chunk(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n, const int chunk_start, const int chunk_size);
}

#endif //KERNELS_H
