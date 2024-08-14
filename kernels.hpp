#ifndef KERNELS_H
#define KERNELS_H

namespace kernels {
    __global__ void rank1_update_means(float* d_mean, const float* d_new_row, const int cols, const int n);
    __global__ void rank1_calculate_covariance(float* d_cov_matrix, const float* d_new_row, const float* d_mean, const int cols, const int n);
    __global__ void rank1_update_kernel_st(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n);
    __global__ void update_means_inplace(const float* d_data, float* d_mean, const int rows, const int cols);
    __global__ void subtract_means(const float* d_data, const float* d_mean, float* d_centered, const int rows, const int cols);
}

#endif //KERNELS_H
