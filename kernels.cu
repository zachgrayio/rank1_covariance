#include "kernels.hpp"

#include <cstdio>

namespace kernels {

    // the simplest possible rank1 update kernel, single threaded and meant to be launched this way with 1 block 1 thread
    __global__ void rank1_update_kernel_st(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n) {
        extern __shared__ float s_mem[];
        float* delta_old = s_mem;
        float* delta_new = s_mem + cols;

        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if (n == 1) {
                for (int i = 0; i < cols; ++i) {
                    d_mean[i] = d_new_row[i];
                }
                for (int i = 0; i < cols; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        d_cov_matrix[i * cols + j] = 0.0f;
                    }
                }
            } else {
                for (int i = 0; i < cols; ++i) {
                    delta_old[i] = d_new_row[i] - d_mean[i];
                    d_mean[i] = d_mean[i] + delta_old[i] / n;
                    delta_new[i] = d_new_row[i] - d_mean[i];
                }

                const float alpha = 1.0f / (n - 1.0f);
                for (int i = 0; i < cols; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        d_cov_matrix[i * cols + j] = (n - 2) / (n - 1.0f) * d_cov_matrix[i * cols + j] + alpha * (delta_old[i] * delta_new[j]);
                    }
                }
            }
        }
    }

    __global__ void rank1_update_means(float* d_mean, const float* d_new_row, const int cols, const int n) {
        if (const int thread_id = threadIdx.x + blockIdx.x * blockDim.x; thread_id < cols) {
            const float delta_old = d_new_row[thread_id] - d_mean[thread_id];
            d_mean[thread_id] = fmaf(delta_old, 1.0f / n, d_mean[thread_id]);
        }
    }

    __global__ void rank1_calculate_covariance(float* d_cov_matrix, const float* d_new_row, const float* d_mean, const int cols, const int n) {
        if (const int thread_id = threadIdx.x + blockIdx.x * blockDim.x; thread_id < cols * cols) {
            const int row = thread_id / cols;
            const int col = thread_id % cols;

            const float delta_old = d_new_row[row] - d_mean[row];
            const float delta_new = d_new_row[col] - d_mean[col];

            const float alpha = 1.0f / (n - 1.0f);
            const float scale = (n - 2) / (n - 1.0f);
            d_cov_matrix[row * cols + col] = fmaf(scale, d_cov_matrix[row * cols + col], alpha * delta_old * delta_new);
        }
    }

    // a helper kernel to update the means in-place as part of centering ahead of sgemm call in the 1shot covariance
    // method
    __global__ void update_means_inplace(const float* d_data, float* d_mean, const int rows, const int cols) {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= cols) return;

        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sum = fmaf(d_data[i * cols + col], 1.0f, sum);
        }
        d_mean[col] = sum / rows;
    }


    // a helper kernel to remove the pre-calculated means as part of centering ahead of sgemm call in the 1
    // shot covariance method
    __global__ void subtract_means(const float* d_data, const float* d_mean, float* d_centered, const int rows, const int cols) {
        if (const int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols) {
            const int col = idx % cols;
            d_centered[idx] = d_data[idx] - d_mean[col];
        }
    }

}
