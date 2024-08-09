import torch
import numpy as np

def calculate_mean(data):
    return torch.mean(data, dim=0)

def subtract_mean(data, mean):
    return data - mean

def calculate_covariances_1shot_mixed_precision(data):
    mean = calculate_mean(data)
    centered_data = subtract_mean(data, mean)
    cov_matrix = torch.mm(centered_data.T, centered_data) / (data.size(0) - 1)
    return cov_matrix, mean

def rank1_update(cov_matrix, new_row, mean, n):
    # calculate the new mean
    new_mean = mean + (new_row - mean) / n
    # update the covariance matrix
    delta_old = new_row - mean
    delta_new = new_row - new_mean
    cov_matrix = ((n - 2) / (n - 1)) * cov_matrix + torch.outer(delta_old, delta_new) / (n - 1)
    return cov_matrix, new_mean

def test_incremental_covariance(data):
    num_updates = data.size(0) - data.size(1)
    order = data.size(1)
    print("initial full matrix calculation (1-shot):")
    cov_matrix_1shot, mean = calculate_covariances_1shot_mixed_precision(data[:order])
    print(f"initial Mean: {mean}")
    print(f"initial covariance matrix:\n{cov_matrix_1shot}\n")

    for i in range(num_updates):
        new_row = data[order + i]
        print(f"update {i+1} with new row {new_row}:")
        cov_matrix_1shot, mean = rank1_update(cov_matrix_1shot, new_row, mean, order + i + 1)
        print(f"updated Mean: {mean}")
        print(f"updated covariance matrix:\n{cov_matrix_1shot}\n")
        # 1-shot covariance matrix to ensure incremental update was correct
        cov_matrix_1shot_check, _ = calculate_covariances_1shot_mixed_precision(data[:order + i + 1])
        print(f"1-shot covariance matrix as check for update {i+1}:\n{cov_matrix_1shot_check}\n")

def main():
    # sanity check with np
    data = np.array([
        [0.10, 0.10, 0.10, 0.10],
        [0.20, 0.20, 0.20, 0.20],
        [0.30, 0.30, 0.30, 0.30],
        [0.40, 0.40, 0.40, 0.40]
    ])
    cov_matrix = np.cov(data, rowvar=False)
    print("numpy covariance matrix:\n", cov_matrix)
    # torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.tensor([
        [0.10, 0.10, 0.10, 0.10],
        [0.20, 0.20, 0.20, 0.20],
        [0.30, 0.30, 0.30, 0.30],
        [0.40, 0.40, 0.40, 0.40],
        [0.22, 0.23, 0.24, 0.25],
        [0.26, 0.27, 0.28, 0.29],
        [0.30, 0.31, 0.32, 0.33]
    ], device=device)
    test_incremental_covariance(data)

if __name__ == "__main__":
    main()
