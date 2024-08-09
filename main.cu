#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#define VERBOSE

inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void log(const char* s) {
#ifdef VERBOSE
    std::cout << s << "\n";
#endif
}

void print_matrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

__global__ void calculate_mean(const float* d_data, float* d_mean, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;

    float sum = 0.0f;
    for (int i = 0; i < rows; ++i) {
        sum = fmaf(d_data[i * cols + col], 1.0f, sum);  // Fused multiply-add
    }
    d_mean[col] = sum / rows;
}

__global__ void subtract_mean(const float* d_data, const float* d_mean, float* d_centered_data, int rows, int cols) {
    if (const int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols) {
        const int col = idx % cols;
        d_centered_data[idx] = d_data[idx] - d_mean[col];
    }
}

void calculate_covariances_1shot(const float* d_data, float* d_cov_matrix, float* d_mean, const int rows, const int cols) {
    float* d_centered_data;
    checkCudaErrors(cudaMalloc(&d_centered_data, rows * cols * sizeof(float)));

    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    calculate_mean<<<blocks, threads>>>(d_data, d_mean, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize());

    subtract_mean<<<blocks, threads>>>(d_data, d_mean, d_centered_data, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize());

    cublasHandle_t handle;
    cublasCreate(&handle);
    // the alpha here is how we apply the "over n - 1" part of the formula directly in the GEMM
    // so we don't need to scale it as an additional op
    const float alpha = 1.0f / (rows - 1);
    constexpr float beta = 0.0f;
    cublasSgemm_v2(handle,
                   CUBLAS_OP_N, CUBLAS_OP_T,
                   cols, cols, rows,
                   &alpha,
                   d_centered_data, cols,
                   d_centered_data, cols,
                   &beta,
                   d_cov_matrix, cols);

    cublasDestroy(handle);
    cudaFree(d_centered_data);
}

__global__ void rank1_update_kernel(float* d_cov_matrix, const float* d_new_row, float* d_mean, int cols, int n) {
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

void calculate_covariances_rank1(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n) {
    int threads = cols;
    int blocks = 1;

    rank1_update_kernel<<<blocks, threads>>>(d_cov_matrix, d_new_row, d_mean, cols, n);
    checkCudaErrors(cudaDeviceSynchronize());
}

void test_incremental_covariance() {
    constexpr int rows = 7;
    int cols = 4;
    constexpr float h_data[] = {
        0.10f, 0.10f, 0.10f, 0.10f,
        0.20f, 0.20f, 0.20f, 0.20f,
        0.30f, 0.30f, 0.30f, 0.30f,
        0.40f, 0.40f, 0.40f, 0.40f,
        0.22f, 0.23f, 0.24f, 0.25f,
        0.26f, 0.27f, 0.28f, 0.29f,
        0.30f, 0.31f, 0.32f, 0.33f
    };
    float* d_data;
    checkCudaErrors(cudaMalloc(&d_data, rows * cols * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    float* d_cov_matrix;
    float* d_mean;
    checkCudaErrors(cudaMalloc(&d_cov_matrix, cols * cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean, cols * sizeof(float)));

    calculate_covariances_1shot(d_data, d_cov_matrix, d_mean, cols, cols);

#ifdef VERBOSE
    float h_mean[cols];
    float h_cov_matrix[cols * cols];

    checkCudaErrors(cudaMemcpy(h_mean, d_mean, cols * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "initial means: ";
    for (int i = 0; i < cols; ++i) {
        std::cout << std::fixed << std::setprecision(4) << h_mean[i] << " ";
    }
    std::cout << std::endl << "initial covariance matrix:" << std::endl;
    print_matrix(h_cov_matrix, cols, cols);
    std::cout << "\n";
#endif

    for (int i = cols; i < rows; ++i) {
#ifdef VERBOSE
        std::cout << "update " << (i - cols + 1) << " with new row ";
        float new_row[cols];
        checkCudaErrors(cudaMemcpy(new_row, d_data + i * cols, cols * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << new_row[j] << " ";
        }
        std::cout << ":" << std::endl;
#endif
        calculate_covariances_rank1(d_cov_matrix, d_data + i * cols, d_mean, cols, i + 1);

#ifdef VERBOSE
        float h_mean[cols];
        float h_cov_matrix[cols * cols];
        checkCudaErrors(cudaMemcpy(h_mean, d_mean, cols * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "updated means: ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << h_mean[j] << " ";
        }
        std::cout << std::endl << "updated covariance matrix:" << std::endl;
        print_matrix(h_cov_matrix, cols, cols);
        std::cout << "\n";

        // 1-shot covmat as sanity check for each iteration
        std::cout << "1-shot covariance matrix as check for update " << (i - cols + 1) << ":" << std::endl;
        calculate_covariances_1shot(d_data, d_cov_matrix, d_mean, i + 1, cols);

        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
        print_matrix(h_cov_matrix, cols, cols);

        std::cout << "\n===\n";
#endif
    }

    cudaFree(d_cov_matrix);
    cudaFree(d_mean);
}

void test_large_incremental_covariance(const int order, const int num_updates) {
    const auto start = std::chrono::high_resolution_clock::now();

    const int rows = order + num_updates;
    const int cols = order;

    const auto h_data = new float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_data;
    checkCudaErrors(cudaMalloc(&d_data, rows * cols * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    float* d_cov_matrix;
    float* d_mean;
    checkCudaErrors(cudaMalloc(&d_cov_matrix, cols * cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean, cols * sizeof(float)));

    calculate_covariances_1shot(d_data, d_cov_matrix, d_mean, order, cols);

    float* h_cov_matrix = new float[cols * cols];

    for (int i = 0; i < num_updates; ++i) {
        calculate_covariances_rank1(d_cov_matrix, d_data + (order + i) * cols, d_mean, cols, order + i + 1);

        if (i % 100 == 0) {
#ifdef VERBOSE
            std::cout << "verifying update " << i + 1 << std::endl;
#endif
            calculate_covariances_1shot(d_data, d_cov_matrix, d_mean, order + i + 1, cols);

            const auto h_cov_check = new float[cols * cols];
            checkCudaErrors(cudaMemcpy(h_cov_check, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

            // compare h_cov_check with the current covariance matrix
            checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
            bool valid = true;
            for (int j = 0; j < cols * cols; ++j) {
                if (abs(h_cov_matrix[j] - h_cov_check[j]) > 1e-5) {
                    std::cout << "mismatch at element " << j << ": " << h_cov_matrix[j] << " vs " << h_cov_check[j] << std::endl;
                    valid = false;
                    break;
                }
            }
            if (valid) {
#ifdef VERBOSE
                std::cout << "update " << i + 1 << " verification passed.\n";
#endif
            } else {
                std::cout << "update " << i + 1 << " verification failed.\n";
            }
            delete[] h_cov_check;
        }
    }

    delete[] h_data;
    delete[] h_cov_matrix;
    cudaFree(d_data);
    cudaFree(d_cov_matrix);
    cudaFree(d_mean);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> duration = end - start;
    std::cout << "large random test completed in " << duration.count() << " seconds.\n";
}

std::chrono::duration<double> perform_timed_update(
    const bool rank1,
    const float * d_data,
    float *d_cov_matrix,
    float* d_mean,
    float *h_cov_matrix,
    const int n,
    const int cols)
{
    const auto start = std::chrono::high_resolution_clock::now();

    // calculate the new covariance matrix for the square matrix of data with both approaches
    if(rank1) {
        calculate_covariances_rank1(d_cov_matrix, d_data + (n-1) * cols, d_mean, cols, n);
    } else {
        // important note: for some domains, the 1shot method, since it can be called for a subset of data, is likely
        // to make use of a subset of columns--a luxury the incremental approach doesn't have, so here we use cols/2
        // as a reasonable estimate of the reduced work this function would do for a more fair and realistic benchmark
        calculate_covariances_1shot(d_data, d_cov_matrix, d_mean, n, cols/2);
    }
    // copy back to host so this is somewhat realistic
    checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    return elapsed;
}

void compare_speeds(const int order, const int num_updates) {
    const int rows = order + num_updates;
    const int cols = order;

    // generate all the data on the host
    const auto h_data = new float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // copy all data to device; may want to revisit this to craft a realistic benchmark that includes
    // the time it takes to move new data into the device, as real world applications probably will do that each pass
    float* d_data;
    checkCudaErrors(cudaMalloc(&d_data, rows * cols * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    // allocate on the device for initial calculations and fire off the initial first 1shot calculation
    float* d_cov_matrix;
    float* d_mean;
    checkCudaErrors(cudaMalloc(&d_cov_matrix, cols * cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean, cols * sizeof(float)));
    calculate_covariances_1shot(d_data, d_cov_matrix, d_mean, order, cols);

    // perform the incremental updates and accumulate times;
    std::chrono::duration<double> total_time_rank1{0};
    const auto h_cov_matrix = new float[cols * cols];
    for (int i = 0; i < num_updates; ++i) {
        total_time_rank1 += perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + i + 1, cols);
    }

    std::chrono::duration<double> total_time_1shot{0};
    for (int i = 0; i < num_updates; ++i) {
        total_time_1shot += perform_timed_update(false, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + i + 1, cols);
    }

    // summarize results
    std::cout << "total time for rank1 update: " << total_time_rank1.count() << " seconds" << std::endl;
    std::cout << "total time for 1shot update: " << total_time_1shot.count() << " seconds" << std::endl;
    double iterations_per_second_rank1 = static_cast<double>(num_updates) / total_time_rank1.count();
    double iterations_per_second_1shot = static_cast<double>(num_updates) / total_time_1shot.count();
    std::cout << "iterations per second for rank1 update: " << iterations_per_second_rank1 << " iterations/second" << std::endl;
    std::cout << "iterations per second for 1shot update: " << iterations_per_second_1shot << " iterations/second" <<std::endl;
    
    delete[] h_data;
    delete[] h_cov_matrix;
    cudaFree(d_data);
    cudaFree(d_cov_matrix);
    cudaFree(d_mean);
}

int main(const int argc, char** argv) {
    // simple small test using static data, if VERBOSE is defined it will print detailed
    // covariance matrices
    test_incremental_covariance();

    // the larger tests are parameterized and generate random data for testing
    int order = 5000;
    int num_updates = 2520;
    if (argc > 1) {
        order = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_updates = std::atoi(argv[2]);
    }
    
    test_large_incremental_covariance(order, num_updates);

    compare_speeds(order, num_updates);

    return 0;
}
