# rank-1 covariance matrix calculation

Within this repo there are 2 simple programs that calculate covariances for a set of inputs on the GPU. 

Each program calculates these matrices either in the simple, traditional manner, or in an "incremental" or "rank-1" mode
where _only_ the latest row included in the calculation is factored in as it appears, leading to large performance gains
over a traditional matmul based approach.

- A Python reference implementation: [main.py](main.py)
  - PyTorch + CUDA if available
  - also uses `numpy`'s `.cov()` for a simple sanity check:
- A CUDA C++ binary [main.cu](main.cu)
  - includes a custom rank1 kernel
  - also a "1shot" covariance function that does roughly the same as `.cov` AFAIK
    - makes use of cuBLAS's `cublasSgemm_v2` and a few small kernels to speed up taking col means and centering
  - test coverage to ensure the results are correct
  - a comparison function that compares timing of the 2 approaches

## Results on my RTX 3080 16Gi
```text
...
total time for rank1 update: 20.0093 seconds
total time for 1shot update: 28.7502 seconds
iterations per second for rank1 update: 125.9415 iterations/second
iterations per second for 1shot update: 87.6515 iterations/second
```

## Run it yourself

### Requirements
Requires NVIDIA hardware of course as well as a working CUDA setup but edit `CMakeLists.txt` as needed if it doesn't work
with your setup. Python code should be portable enough to run anywhere, if you have `torch` and `numpy` present.

### Running
To run everything:

```bash
make
```

For advanced usage, see the [Makefile](Makefile).

## Profiling
I haven't had a chance to profile this yet, there's certainly a few bottlenecks to chase down.

## WTF is Rank-1?
A rank-1 update in linear algebra refers to the process of modifying a matrix by adding or subtracting a matrix that can be expressed as the outer product of two vectors.
Specifically, if you have an existing matrix $A$, a rank-1 update to this matrix would look like:

$A' = A + u \cdot v^T$

where $u$ and $v$ are column vectors, and $v^T$ is the transpose of $v$. The product $u \cdot v^T$ results in a matrix 
of the same dimensions as $A$, but it is a rank-1 matrix because it is formed by the outer product of two vectors.

This operation is called a rank-1 update because it increases (or decreases) the rank of the matrix by at most 1.

### How that is applied in these 2 programs

In the context of calculating a covariance matrix, a rank-1 update is particularly useful when you have a sequence of data points arriving one at a time, and you want to update the covariance matrix incrementally without recalculating it from scratch.

Given a covariance matrix $\Sigma$, a new data point $x_{\text{new}}$, and the current mean vector $\mu$ based on $n-1$ samples, the rank-1 update can be mathematically described as follows:

1. **Calculate the new mean**:
  
   $\mu_{\text{new}} = \mu + \frac{x_{\text{new}} - \mu}{n}$

2. **Compute the deviation vectors**:
   
    $\delta_{\text{old}} = x_{\text{new}} - \mu$

    $\delta_{\text{new}} = x_{\text{new}} - \mu_{\text{new}}$

3. **Update the covariance matrix**:

   $\Sigma_{\text{new}} = \frac{n-2}{n-1} \Sigma + \frac{\delta_{\text{old}} \cdot \delta_{\text{new}}^T}{n-1}$

This update rule efficiently incorporates the new data point into the existing covariance matrix $\Sigma$ by adjusting it with the outer product of the deviation vectors $\delta_{\text{old}}$ and $\delta_{\text{new}}$.

### In practice

Updating a covariance matrix with a new row in this case is as simple as the following, taken directly from the python reference impl:

```python
def rank1_update(cov_matrix, new_row, mean, n):
    # calculate the new mean
    new_mean = mean + (new_row - mean) / n
    # update the covariance matrix
    delta_old = new_row - mean
    delta_new = new_row - new_mean
    cov_matrix = ((n - 2) / (n - 1)) * cov_matrix + torch.outer(delta_old, delta_new) / (n - 1)
    return cov_matrix, new_mean
```

and while it's a bit more noisy, the CUDA kernel bears a lot of resemblance:

```c++
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
```