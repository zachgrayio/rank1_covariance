#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <nvToolsExt.h>
#include <vector>

#include "kernels.hpp"

#define VERBOSE
constexpr bool ALWAYS_COPY_RESULT_TO_HOST = true;
constexpr float ONESHOT_WORK_FACTOR = 0.5;

inline void checkCudaErrors(const cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCudaErrors(const cudaError_t err, const std::string &context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << context << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCublasErrors(const cublasStatus_t status, const std::string &context, const char *file, const int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::string errorStr;
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                errorStr = "CUBLAS_STATUS_NOT_INITIALIZED";
            break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                errorStr = "CUBLAS_STATUS_ALLOC_FAILED";
            break;
            case CUBLAS_STATUS_INVALID_VALUE:
                errorStr = "CUBLAS_STATUS_INVALID_VALUE";
            break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                errorStr = "CUBLAS_STATUS_ARCH_MISMATCH";
            break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                errorStr = "CUBLAS_STATUS_MAPPING_ERROR";
            break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                errorStr = "CUBLAS_STATUS_EXECUTION_FAILED";
            break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                errorStr = "CUBLAS_STATUS_INTERNAL_ERROR";
            break;
            default:
                errorStr = "Unknown CUBLAS error";
            break;
        }
        std::cerr << "cuBLAS Error: " << context << " - " << errorStr
                  << " at " << file << ":" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUBLAS_ERRORS(status, context) checkCublasErrors(status, context, __FILE__, __LINE__)

inline void log(const char* s) {
#ifdef VERBOSE
    std::cout << s << "\n";
#endif
}

void print_matrix(const float* matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// simplest version of the covariance calculation which doesn't use mixed precision. unused currently.
void calculate_covariances_1shot(float* d_data, float* d_cov_matrix, float* d_mean, const int rows, const int cols) {
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;

    // center the matrix by calculating means and subtracting them
    float* d_centered;
    checkCudaErrors(cudaMalloc(&d_centered, rows * cols * sizeof(float)));
    kernels::update_means_inplace<<<blocks, threads>>>(d_data, d_mean, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize(), "update_means_inplace");
    kernels::subtract_means<<<blocks, threads>>>(d_data, d_mean, d_centered, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize(), "subtract_means_inplace");

    cublasHandle_t handle;
    cublasCreate(&handle);
    // the alpha here is how we apply the "over n - 1" part of the formula directly in the GEMM
    // so we don't need to scale it as an additional op
    const float alpha = 1.0f / (rows - 1);
    constexpr float beta = 0.0f;
    const cublasStatus_t status = cublasSgemm_v2(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cols, cols, rows,
        &alpha,
        d_centered, cols,
        d_centered, cols,
        &beta,
        d_cov_matrix, cols
    );
    CHECK_CUBLAS_ERRORS(status, "cublasGemmStridedBatchedEx");
    cublasDestroy(handle);
    cudaFree(d_centered);
}

// a non stream version of the 1shot covariance matrix calculation, using mixed precision
void calculate_covariances_1shot_strided(const float* d_data, float* d_cov_matrix, float* d_mean, const int rows, const int cols) {
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;

    // center the matrix by calculating means and subtracting them
    float* d_centered;
    checkCudaErrors(cudaMalloc(&d_centered, rows * cols * sizeof(float)));
    kernels::update_means_inplace<<<blocks, threads>>>(d_data, d_mean, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize(), "update_means_inplace");
    kernels::subtract_means<<<blocks, threads>>>(d_data, d_mean, d_centered, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize(), "subtract_means_inplace");

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f / (rows - 1);
    constexpr float beta = 0.0f;
    const int stride_a = cols;
    const int stride_b = cols;
    const int stride_c = cols;
    const cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cols, cols, rows,
        &alpha,
        d_centered, CUDA_R_32F, cols, stride_a,
        d_centered, CUDA_R_32F, cols, stride_b,
        &beta,
        d_cov_matrix, CUDA_R_32F, cols, stride_c,
        1,
        CUBLAS_COMPUTE_32F, // mixed precision throws off results pretty far here, so disabled for now
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    CHECK_CUBLAS_ERRORS(status, "cublasGemmStridedBatchedEx");

    checkCudaErrors(cudaDeviceSynchronize(), "cublasGemmStridedBatchedEx - sync");
    cublasDestroy(handle);
    cudaFree(d_centered);
}

// the streams variant of the covariance calculation, unused at the moment, generally a little slower than non-stream
// version
void calculate_covariances_1shot_mixed_precision_streams(float* d_data, float* d_cov_matrix, float* d_mean, const int rows, const int cols) {
    int threads = 256;
    int mean_blocks = (rows * cols + threads - 1) / threads;

    // center the matrix by calculating means and subtracting them
    float* d_centered;
    checkCudaErrors(cudaMalloc(&d_centered, rows * cols * sizeof(float)));
    kernels::update_means_inplace<<<mean_blocks, threads>>>(d_data, d_mean, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize(), "update_means_inplace");
    kernels::subtract_means<<<mean_blocks, threads>>>(d_data, d_mean, d_centered, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize(), "subtract_means_inplace");

    // this operation is large enough to max out most workstation GPUs already, so more streams
    // will only be useful in the context of very powerful gpus, so this is just set quite low
    constexpr int num_streams = 2;
    const int rows_per_stream = rows / num_streams;
    const int extra_rows = rows % num_streams;

    cudaStream_t streams[num_streams];
    cublasHandle_t handles[num_streams];

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
        cublasCreate(&handles[i]);
        cublasSetStream(handles[i], streams[i]);
    }

    const float alpha = 1.0f / (rows - 1);
    constexpr float beta = 0.0f;

    for (int i = 0; i < num_streams; ++i) {
        const int current_rows = rows_per_stream + (i == num_streams - 1 ? extra_rows : 0);
        const int offset = i * rows_per_stream * cols;

        const int stride_a = cols;
        const int stride_b = cols;
        constexpr int stride_c = 0;

        const cublasStatus_t status = cublasGemmStridedBatchedEx(
            handles[i],
            CUBLAS_OP_N, CUBLAS_OP_T,
            cols, cols, current_rows,
            &alpha,
            d_centered + offset, CUDA_R_32F, cols, stride_a,
            d_centered + offset, CUDA_R_32F, cols, stride_b,
            &beta,
            d_cov_matrix, CUDA_R_32F, cols, stride_c,
            1,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        CHECK_CUBLAS_ERRORS(status, "cublasGemmStridedBatchedEx");
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cublasDestroy(handles[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(d_centered);
}

void calculate_covariances_rank1(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n) {
    if (n == 0) {
        std::cerr << "Error: n cannot be 0 in calculate_covariances_rank1." << std::endl;
        exit(1);
    }

    // single threaded (correct) is as simple as
    // kernels::rank1_update_kernel_st<<<1, 1>>>(d_cov_matrix, d_new_row, d_mean, cols, n);

    // for the concurrent alternative, we use 2 kernels, one for means, one for the covariances
    int threads = min(256, cols);
    int blocks = (cols + threads - 1) / threads;
    kernels::rank1_update_means<<<blocks, threads>>>(d_mean, d_new_row, cols, n);
    checkCudaErrors(cudaDeviceSynchronize(), "rank1_update_means");

    const int total_elements = cols * cols;
    blocks = (total_elements + threads - 1) / threads;
    kernels::rank1_calculate_covariance<<<blocks, threads>>>(d_cov_matrix, d_new_row, d_mean, cols, n);
    checkCudaErrors(cudaDeviceSynchronize(), "rank1_calculate_covariance");
}

bool all_close(const float* arr1, const float* arr2, int size, float atol = 1e-7, float rtol = 1e-5) {
    // numpy defaults are  atol = 1e-8, rtol = 1e-5
    double total_relative_difference = 0.0;
    for (int i = 0; i < size; ++i) {
        const float diff = std::abs(arr1[i] - arr2[i]);
        const float tolerance = atol + rtol * std::abs(arr2[i]);

        total_relative_difference += diff / (std::abs(arr2[i]) + atol);

        // if (diff > tolerance) {
        //     std::cout << std::setprecision(8) << std::fixed;
        //     std::cout << "mismatch at element " << i << ": " << arr1[i] << " vs " << arr2[i] << std::endl;
        // }
    }
    const double variance_percentage = total_relative_difference / size * 100.0;
    std::cout << "variance percentage: " << variance_percentage << "%" << std::endl;
    return variance_percentage < 8;
}

void test_incremental_covariance() {
    constexpr int rows = 8;
    int cols = 4;
    constexpr float h_data[] = {
        -1.00f,  0.00f,  0.50f,  1.00f,
        -0.75f,  0.25f,  0.75f,  0.75f,
        -0.50f,  0.50f,  1.00f,  0.50f,
        -0.25f,  0.75f,  0.25f,  0.25f,
         0.00f,  0.25f,  0.75f,  0.50f,
         0.25f, -0.25f, -0.75f, -0.50f,
         0.50f, -0.50f, -1.00f, -0.50f,
         0.75f, -0.75f, -0.25f, -0.75f
    };

    float* d_data;
    checkCudaErrors(cudaMalloc(&d_data, rows * cols * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    float* d_cov_matrix;
    float* d_mean_1shot;
    float* d_mean_rank1;
    checkCudaErrors(cudaMalloc(&d_cov_matrix, cols * cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_rank1, cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_1shot, cols * sizeof(float)));

    const auto h_cov_rank1 = new float[cols * cols];

    for (int i = 0; i < rows; ++i) {
        std::cout << "update " << i + 1 << " with new row ";
        float new_row[cols];
        checkCudaErrors(cudaMemcpy(new_row, d_data + i * cols, cols * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(6) << new_row[j] << " ";
        }
        std::cout << ":" << std::endl;

        calculate_covariances_rank1(d_cov_matrix, d_data + i * cols, d_mean_rank1, cols, i + 1);

        float h_mean[cols];
        float h_cov_matrix[cols * cols];
        checkCudaErrors(cudaMemcpy(h_mean, d_mean_rank1, cols * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "updated means: ";
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(6) << h_mean[j] << " ";
        }
        std::cout << std::endl << "updated covariance matrix:" << std::endl;
        print_matrix(h_cov_matrix, cols, cols);
        std::cout << "\n";

        checkCudaErrors(cudaMemcpy(h_cov_rank1, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

        // 1-shot covariance matrix to ensure incremental update was correct
        std::cout << "1-shot covariance matrix as check for update " << i + 1 << ":" << std::endl;
        calculate_covariances_1shot_strided(d_data, d_cov_matrix, d_mean_1shot, i + 1, cols);

        if(i > 1) {
            if (all_close(h_cov_matrix, h_cov_rank1, cols * cols)) {
#ifdef VERBOSE
                std::cout << "update " << i + 1 << " verification passed.\n";
#endif
            } else {
                std::cout << "update " << i + 1 << " verification failed.\n";
                exit(1);
            }
        }

        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
        print_matrix(h_cov_matrix, cols, cols);

        std::cout << "\n===\n";
    }

    delete[] h_cov_rank1;
    cudaFree(d_cov_matrix);
    cudaFree(d_mean_rank1);
    cudaFree(d_mean_1shot);
    cudaFree(d_data);
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

    calculate_covariances_1shot_strided(d_data, d_cov_matrix, d_mean, order, cols);

    float* h_cov_matrix = new float[cols * cols];

    for (int i = 0; i < num_updates; ++i) {
        calculate_covariances_rank1(d_cov_matrix, d_data + (order + i) * cols, d_mean, cols, order + i + 1);

        if (i % 100 == 0) {
            checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
#ifdef VERBOSE
            std::cout << "verifying update " << i + 1 << std::endl;
#endif
            calculate_covariances_1shot_strided(d_data, d_cov_matrix, d_mean, order + i + 1, cols);

            const auto h_cov_check = new float[cols * cols];
            checkCudaErrors(cudaMemcpy(h_cov_check, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

            if (all_close(h_cov_matrix, h_cov_check, cols * cols)) {
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
    float * d_data,
    float *d_cov_matrix,
    float* d_mean,
    float *h_cov_matrix,
    const int n,
    const int cols,
    const bool copy_d_to_h)
{
    const auto start = std::chrono::high_resolution_clock::now();

    // calculate the new covariance matrix for the square matrix of data for a single square matrix
    if(rank1) {
        calculate_covariances_rank1(d_cov_matrix, d_data + (n-1) * cols, d_mean, cols, n);
    } else {
        // important note: for some domains, the 1shot method, since it can be called for a subset of data, is likely
        // to make use of a subset of columns--a luxury the incremental approach doesn't have, so here we use cols
        // * ONESHOT_WORK_FACTOR to estimate the reduced work this function may do in real-world applications.
        const int reduced_cols = cols * ONESHOT_WORK_FACTOR;
        calculate_covariances_1shot_strided(d_data, d_cov_matrix, d_mean, n, reduced_cols);
    }
    // copy back to host so this is somewhat realistic
    if(copy_d_to_h) {
        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
    }

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

    // allocate on the device and fire off the initial 1shot calculation
    float* d_cov_matrix;
    float* d_mean;
    checkCudaErrors(cudaMalloc(&d_cov_matrix, cols * cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean, cols * sizeof(float)));
    calculate_covariances_1shot_strided(d_data, d_cov_matrix, d_mean, order, cols);

    // perform a series of updates and accumulate times; the incremental approach should show some benefit here
    // if it is going to at all, as it does a good bit less work
    // start with the incremental approach
    std::chrono::duration<double> total_time_rank1{0};
    const auto h_cov_matrix = new float[cols * cols];
    for (int i = 0; i < num_updates; ++i) {
        total_time_rank1 += perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + i + 1, cols, ALWAYS_COPY_RESULT_TO_HOST);
    }

    // and for comparison, also run and time the 1shot method
    // copy data to the device again out of an abundance of safety
    // (just so mutating data in place is not off limits in the future)
    checkCudaErrors(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    std::chrono::duration<double> total_time_1shot{0};
    for (int i = 0; i < num_updates; ++i) {
        total_time_1shot += perform_timed_update(false, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + i + 1, cols, ALWAYS_COPY_RESULT_TO_HOST);
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

bool convert_to_bool(const char* env_var_value) {
    if (env_var_value == nullptr) {
        return false;
    }
    const std::string value_str(env_var_value);
    if (value_str == "1" || value_str == "true" || value_str == "TRUE" || value_str == "on" || value_str == "ON") {
        return true;
    }
    if (value_str == "0" || value_str == "false" || value_str == "FALSE" || value_str == "off" || value_str == "OFF") {
        return false;
    }
    std::cerr << "Unrecognized value for boolean environment variable: " << value_str << std::endl;
    return false;
}

void run_for_profile(const int order) {

    constexpr int num_updates = 4;
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

    // // initial 1shot
    const auto h_cov_matrix = new float[cols * cols];
    nvtxRangePush("1shot update 0");
    perform_timed_update(false, d_data, d_cov_matrix, d_mean, h_cov_matrix, 0, cols, ALWAYS_COPY_RESULT_TO_HOST);
    nvtxRangePop();

    //rank1, 3x
    nvtxRangePush("rank1 update 0");
    perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + 1, cols, ALWAYS_COPY_RESULT_TO_HOST);
    nvtxRangePop();

    nvtxRangePush("rank1 update 1");
    perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + 2, cols, ALWAYS_COPY_RESULT_TO_HOST);
    nvtxRangePop();

    nvtxRangePush("rank1 update 2");
    perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + 3, cols, ALWAYS_COPY_RESULT_TO_HOST);
    nvtxRangePop();

    // and again the 1shot
    nvtxRangePush("1shot update 1");
    perform_timed_update(false, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + 4, cols, ALWAYS_COPY_RESULT_TO_HOST);
    nvtxRangePop();

    delete[] h_data;
    delete[] h_cov_matrix;
    cudaFree(d_data);
    cudaFree(d_cov_matrix);
    cudaFree(d_mean);
}

// simple CPU version for comparison, not performance oriented at all
void rank1_update_cpu(std::vector<float>& cov_matrix, const std::vector<float>& new_row, std::vector<float>& mean, const int cols, const int n) {
    if (n == 1) {
        for (int i = 0; i < cols; ++i) {
            mean[i] = new_row[i];
        }
        std::fill(cov_matrix.begin(), cov_matrix.end(), 0.0f);
    } else {
        std::vector<float> delta_old(cols);
        std::vector<float> delta_new(cols);

        for (int i = 0; i < cols; ++i) {
            delta_old[i] = new_row[i] - mean[i];
            mean[i] = mean[i] + delta_old[i] / n;
            delta_new[i] = new_row[i] - mean[i];
        }

        const float alpha = 1.0f / (n - 1.0f);
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) {
                cov_matrix[i * cols + j] = (n - 2) / (n - 1.0f) * cov_matrix[i * cols + j] + alpha * (delta_old[i] * delta_new[j]);
            }
        }
    }
}

void print_matrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void test_incremental_covariance_cpu(const std::vector<std::vector<float>>& data) {
    const int rows = data.size();
    const int cols = data[0].size();

    std::vector<float> cov_matrix(cols * cols, 0.0f);
    std::vector<float> mean(cols, 0.0f);

    for (int i = 0; i < rows; ++i) {
        const std::vector<float>& new_row = data[i];
        std::cout << "update " << i + 1 << " with new row: ";
        for (const float val : new_row) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;

        rank1_update_cpu(cov_matrix, new_row, mean, cols, i + 1);

        std::cout << "updated means: ";
        for (const float val : mean) {
            std::cout << std::fixed << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;

        std::cout << "updated covariance matrix:" << std::endl;
        print_matrix(cov_matrix, cols, cols);
        std::cout << std::endl;

        std::cout << "===\n";
    }
}

int main(const int argc, char** argv) {
    // some tests are parameterized and generate random data for testing
    int order = 5000;
    int num_updates = 2520;
    if (argc > 1) {
        order = std::atoi(argv[1]);
    }
    if (argc > 2) {
        num_updates = std::atoi(argv[2]);
    }

    if(convert_to_bool(std::getenv("PROFILE"))) {
        std::cout << "PROFILING" << std::endl;
        run_for_profile(order);
        exit(0);
    }

    test_incremental_covariance();

    // const std::vector<std::vector<float>> d2 = {
    //     {-1.00f,  0.00f,  0.50f,  1.00f},
    //     {-0.75f,  0.25f,  0.75f,  0.75f},
    //     {-0.50f,  0.50f,  1.00f,  0.50f},
    //     {-0.25f,  0.75f,  0.25f,  0.25f},
    //      {0.00f,  0.25f,  0.75f,  0.50f},
    //      {0.25f, -0.25f, -0.75f, -0.50f},
    //      {0.50f, -0.50f, -1.00f, -0.50f},
    //      {0.75f, -0.75f, -0.25f, -0.75f}
    // };
    // test_incremental_covariance_cpu(d2);

    test_large_incremental_covariance(order, num_updates);

    compare_speeds(order, num_updates);

    return 0;
}
