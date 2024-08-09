#ifndef KERNELS_H
#define KERNELS_H

namespace kernels {
__global__ void rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n);
__global__ void shared_rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n);
__global__ void update_means_inplace(const float* d_data, float* d_mean, const int rows, const int cols);
__global__ void subtract_means_inplace(float* d_data, const float* d_mean, const int rows, const int cols);
__global__ void update_and_subtract_means_inplace(float* d_data, float* d_mean, const int rows, const int cols);
__global__ void shared_update_and_subtract_means_inplace(float* d_data, const int rows, const int cols);
}

#endif //KERNELS_H
