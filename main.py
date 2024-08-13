import numpy as np
import torch

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
    if n == 1:
        new_mean = new_row
        cov_matrix = torch.zeros_like(cov_matrix)
    else:
        new_mean = mean + (new_row - mean) / n
        delta_old = new_row - mean
        delta_new = new_row - new_mean
        cov_matrix = ((n - 2) / (n - 1)) * cov_matrix + torch.outer(delta_old, delta_new) / (n - 1)
    return cov_matrix, new_mean


def test_incremental_covariance(data):
    order = data.size(1)
    cov_matrix_rank1 = torch.zeros((order, order), device=data.device)
    mean_rank1 = torch.zeros(order, device=data.device)

    for i in range(data.size(0)):
        new_row = data[i]
        print(f"update {i+1} with new row {new_row.cpu().numpy()}:")

        cov_matrix_rank1, mean_rank1 = rank1_update(cov_matrix_rank1, new_row, mean_rank1, i + 1)

        print(f"updated means: {mean_rank1.cpu().numpy()}")
        print(f"updated covariance matrix:\n{cov_matrix_rank1.cpu().numpy()}\n")

        # 1-shot covariance matrix to ensure incremental update was correct
        cov_matrix_1shot_check, _ = calculate_covariances_1shot_mixed_precision(data[:i + 1])
        print(f"1-shot covariance matrix as check for update {i+1}:\n{cov_matrix_1shot_check.cpu().numpy()}\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sanity check with np
    data = np.array([
        [-1.00,  0.00,  0.50,  1.00],
        [-0.75,  0.25,  0.75,  0.75],
        [-0.50,  0.50,  1.00,  0.50],
        [-0.25,  0.75,  0.25,  0.25],
    ])
    cov_matrix = np.cov(data, rowvar=False)
    print("numpy covariance matrix:\n", cov_matrix)
    cov_matrix_1shot_check, _ = calculate_covariances_1shot_mixed_precision(torch.tensor(data, device=device))
    print("1shot covariance matrix:\n", cov_matrix_1shot_check)

    # torch
    data = torch.tensor([
        [-1.00,  0.00,  0.50,  1.00],
        [-0.75,  0.25,  0.75,  0.75],
        [-0.50,  0.50,  1.00,  0.50],
        [-0.25,  0.75,  0.25,  0.25],
        [0.00,  0.25,  0.75,  0.50],
        [0.25, -0.25, -0.75, -0.50],
        [0.50, -0.50, -1.00, -0.50],
        [0.75, -0.75, -0.25, -0.75]
    ], device=device)
    test_incremental_covariance(data)

if __name__ == "__main__":
    main()
