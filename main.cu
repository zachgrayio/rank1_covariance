#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <nvToolsExt.h>

#include "kernels.hpp"

#define VERBOSE

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

    // this variant combines these ops to a single kernel but this is slower so far in my testing, the kernel
    // needs profiled.
    //
    // kernels::update_and_subtract_means_inplace<<<blocks, threads>>>(d_data, d_mean, rows, cols);
    // checkCudaErrors(cudaDeviceSynchronize(), "update_and_subtract_means_inplace");

    // another variant of centering using shared memory, but so far its only slower, again it needs profiled and launch
    // params tuned.
    //
    // size_t sharedMemSize = threads * sizeof(float);
    // kernels::shared_update_and_subtract_means_inplace<<<blocks, threads, sharedMemSize>>>(d_data, d_mean, rows, cols);
    // checkCudaErrors(cudaDeviceSynchronize(), "shared_update_and_subtract_means_inplace");

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
void calculate_covariances_1shot_mixed_precision(const float* d_data, float* d_cov_matrix, float* d_mean, const int rows, const int cols) {
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
        CUBLAS_COMPUTE_32F_FAST_16F,
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
            CUBLAS_COMPUTE_32F_FAST_16F,
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
    int threads = cols;
    int blocks = 1;

    // to launch the kernel alternative without shared memory
    // rank1_update_kernel<<<blocks, threads>>>(d_cov_matrix, d_new_row, d_mean, cols, n);

    size_t sharedMemSize = 2 * cols * sizeof(float);
    kernels::shared_rank1_update_kernel<<<blocks, threads, sharedMemSize>>>(d_cov_matrix, d_new_row, d_mean, cols, n);

    checkCudaErrors(cudaDeviceSynchronize(), "shared_rank1_update_kernel");
}

// a variant that spawns concurrent kernels to process chunks of big rows in parallel, might be useful on bigger GPUs
void calculate_covariances_rank1_concurrent_chunks(float* d_cov_matrix, const float* d_new_row, float* d_mean, const int cols, const int n) {
    int threads = cols;
    constexpr int blocks = 24;
    constexpr int chunks = 32;
    const int chunk_size = (cols + chunks - 1) / chunks;

    for (int chunk_start = 0; chunk_start < cols; chunk_start += chunk_size) {
        kernels::update_covariance_chunk<<<blocks, threads>>>(d_cov_matrix, d_new_row, d_mean, cols, n, chunk_start, chunk_size);
    }

    checkCudaErrors(cudaDeviceSynchronize(), "update_covariance_chunk");
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

    calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, cols, cols);

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
        calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, i + 1, cols);

        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));
        print_matrix(h_cov_matrix, cols, cols);

        std::cout << "\n===\n";
#endif
    }

    cudaFree(d_cov_matrix);
    cudaFree(d_mean);
}

bool all_close(const float* arr1, const float* arr2, int size, float atol = 1e-3, float rtol = 1e-7) {
    // numpy defaults are  atol = 1e-8, rtol = 1e-5
    for (int i = 0; i < size; ++i) {
        if (std::abs(arr1[i] - arr2[i]) > (atol + rtol * std::abs(arr2[i]))) {
            std::cout << "mismatch at element " << i << ": " << arr1[i] << " vs " << arr2[i] << std::endl;
            return false;
        }
    }
    return true;
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

    calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, order, cols);

    float* h_cov_matrix = new float[cols * cols];

    for (int i = 0; i < num_updates; ++i) {
        calculate_covariances_rank1(d_cov_matrix, d_data + (order + i) * cols, d_mean, cols, order + i + 1);
        checkCudaErrors(cudaMemcpy(h_cov_matrix, d_cov_matrix, cols * cols * sizeof(float), cudaMemcpyDeviceToHost));

        if (i % 100 == 0) {
#ifdef VERBOSE
            std::cout << "verifying update " << i + 1 << std::endl;
#endif
            calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, order + i + 1, cols);

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
    const int cols)
{
    const auto start = std::chrono::high_resolution_clock::now();

    // calculate the new covariance matrix for the square matrix of data for a single square matrix
    if(rank1) {
        calculate_covariances_rank1(d_cov_matrix, d_data + (n-1) * cols, d_mean, cols, n);
    } else {
        // important note: for some domains, the 1shot method, since it can be called for a subset of data, is likely
        // to make use of a subset of columns--a luxury the incremental approach doesn't have, so here we use cols/2
        // as a reasonable estimate of the reduced work this function would do for a more fair and realistic benchmark
        calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, n, cols/2);
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

    // allocate on the device and fire off the initial 1shot calculation
    float* d_cov_matrix;
    float* d_mean;
    checkCudaErrors(cudaMalloc(&d_cov_matrix, cols * cols * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean, cols * sizeof(float)));
    calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, order, cols);

    // perform a series of updates and accumulate times; the incremental approach should show some benefit here
    // if it is going to at all, as it does a good bit less work
    // start with the incremental approach
    std::chrono::duration<double> total_time_rank1{0};
    const auto h_cov_matrix = new float[cols * cols];
    for (int i = 0; i < num_updates; ++i) {
        total_time_rank1 += perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + i + 1, cols);
    }

    // and for comparison, also run and time the 1shot method
    // copy data to the device again out of an abundance of safety
    // (just so mutating data in place is not off limits in the future)
    checkCudaErrors(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
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

    constexpr int num_updates = 1;
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
    // calculate_covariances_1shot_mixed_precision(d_data, d_cov_matrix, d_mean, order, cols);
    // // rank1, once
    const auto h_cov_matrix = new float[cols * cols];
    nvtxRangePush("perform_timed_update 0");
    perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, 0, cols);
    nvtxRangePop();

    nvtxRangePush("perform_timed_update 1");
    perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, 1, cols);
    nvtxRangePop();

    nvtxRangePush("perform_timed_update 2");
    perform_timed_update(true, d_data, d_cov_matrix, d_mean, h_cov_matrix, 2, cols);
    nvtxRangePop();
    // // and again the 1shot
    // perform_timed_update(false, d_data, d_cov_matrix, d_mean, h_cov_matrix, order + num_updates + 1, cols);

    delete[] h_data;
    delete[] h_cov_matrix;
    cudaFree(d_data);
    cudaFree(d_cov_matrix);
    cudaFree(d_mean);
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

    // simple small test using static data, if VERBOSE is defined it will print detailed
    // covariance matrices
    test_incremental_covariance();

    test_large_incremental_covariance(order, num_updates);

    compare_speeds(order, num_updates);

    return 0;
}
