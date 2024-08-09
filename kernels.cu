#include "kernels.hpp"

namespace kernels {

// the simplest rank1 update kernel
__global__ void rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cols) return;

    const float new_mean = fmaf(d_new_row[idx] - d_mean[idx], 1.0f / n, d_mean[idx]);

    // update the covariance matrix
    const int row_start = idx * cols;
    for (int j = 0; j < cols; ++j) {
        const float delta_old = d_new_row[j] - d_mean[j];
        const float delta_new = d_new_row[j] - new_mean;
        d_cov_matrix[row_start + j] = fmaf(delta_old, delta_new / (n - 1), (n - 2) / static_cast<float>(n - 1) * d_cov_matrix[row_start + j]);
    }

    d_mean[idx] = new_mean;
}

// a bit more optimized using shared memory
__global__ void shared_rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n) {
    extern __shared__ float s_mem[];
    float* s_mean = s_mem;
    float* s_new_row = s_mem + cols;
    const int idx = threadIdx.x;

    if (idx < cols) {
        s_mean[idx] = d_mean[idx];
        s_new_row[idx] = d_new_row[idx];
    }
    __syncthreads();

    if (idx < cols) {
        const float new_mean = fmaf(s_new_row[idx] - s_mean[idx], 1.0f / n, s_mean[idx]);
        const int row_start = idx * cols;
        for (int j = 0; j < cols; ++j) {
            const float delta_old = s_new_row[j] - s_mean[j];
            const float delta_new = s_new_row[j] - new_mean;
            d_cov_matrix[row_start + j] = fmaf(delta_old, delta_new / (n - 1), (n - 2) / static_cast<float>(n - 1) * d_cov_matrix[row_start + j]);
        }
        d_mean[idx] = new_mean;
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

// a helper kernel to remove the pre-calculated means in-place as part of centering ahead of sgemm call in the 1
// shot covariance method
__global__ void subtract_means(const float* d_data, const float* d_mean, float* d_centered, const int rows, const int cols) {
    if (const int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols) {
        const int col = idx % cols;
        d_centered[idx] = d_data[idx] - d_mean[col];
    }
}

}
