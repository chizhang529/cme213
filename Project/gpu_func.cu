#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE 32
#define NUM_THREADS 256
#define MAX_NUM_BLOCK 65535

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    return result;
}

__global__
void gpu_softmax_kernel(double *mat, const int M, const int N) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        double sum = 0.0;
        // iterate all classes
        for (size_t c = 0; c < M; ++c) {
            const unsigned int index = M * col + c;
            mat[index] = std::exp(mat[index]);
            sum += mat[index];
        }
        for (size_t c = 0; c < M; ++c) {
            const unsigned int index = M * col + c;
            mat[index] /= sum;
        }
    }
}

void gpu_softmax(double *mat, const int M, const int N) {
    dim3 block(BLOCK_SIZE);

    const unsigned int grid_x = ceil(N / (float)block.x);
    dim3 grid(grid_x);

    gpu_softmax_kernel<<<grid, block>>>(mat, M, N);
};

// __global__
// void gpu_transpose_kernel(double *mat1, double *mat2, int M, int N) {
//     const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
//     const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
//     if (row < M && col < N) {
//         const unsigned int index = M * col + row;
//         const unsigned int new_index = N * row + col;
//         mat2[new_index] = mat1[index];
//     }
// }

// void gpu_transpose(double *mat1, double *mat2, int M, int N) {
//     dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

//     const unsigned int grid_x = ceil(M / (float)block.x);
//     const unsigned int grid_y = ceil(N / (float)block.y);
//     dim3 grid(grid_x, grid_y);

//     gpu_transpose_kernel<<<grid, block>>>(mat1, mat2, M, N);
// }

__global__
void gpu_linear_kernel(const double* __restrict__ mat1, const double* __restrict__ mat2,
                       double* __restrict__ mat3, const double alpha, const double beta,
                       const int M, const int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        mat3[index] = alpha * mat1[index] + beta * mat2[index];
    }
}

void gpu_linear(double *mat1, double *mat2, double *mat3,
                const double alpha, const double beta, const int M, const int N) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_linear_kernel<<<grid, block>>>(mat1, mat2, mat3, alpha, beta, M, N);
}

__global__
void gpu_one_minus_kernel(const double* __restrict__ mat1, double* __restrict__ mat2,
                          const int M, const int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        mat2[index] = 1.0 - mat1[index];
    }
}

void gpu_one_minus(double *mat1, double *mat2, const int M, const int N) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_one_minus_kernel<<<grid, block>>>(mat1, mat2, M, N);
}

__global__
void gpu_row_sum_kernel(const double* __restrict__ mat1, double* __restrict__ mat2,
                        const int M, const int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        double sum = 0.0;
        for (size_t col = 0; col < N; ++col) {
            sum += mat1[M * col + row];
        }
        mat2[row] = sum;
    }
}

void gpu_row_sum(double *mat1, double *mat2, const int M, const int N) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(ceil(N / (float)block.x));
    gpu_row_sum_kernel<<<grid, block>>>(mat1, mat2, M, N);
}

__global__
void gpu_elem_mult_kernel(const double* __restrict__ mat1, double* __restrict__ mat2,
                          const double alpha, const int M, const int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        mat2[index] *= alpha * mat1[index];
    }
}

void gpu_elem_mult(double* mat1, double* mat2, const double alpha, const int M, const int N) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_elem_mult_kernel<<<grid, block>>>(mat1, mat2, alpha, M, N);
}

__global__
void gpu_GEMMSigmoid(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                     const double alpha, const double beta, const int M, const int N, const int K) {
    // thread row and column within Csub
    const unsigned int row = threadIdx.x;
    const unsigned int col = threadIdx.y;
    // index within grid
    const unsigned int grid_row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_col = blockIdx.y * blockDim.y + threadIdx.y;

    double value = 0.0;
    const unsigned int iter = ceil(K / float(BLOCK_SIZE));
    for (int i = 0; i < iter; ++i) {
        // shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];
        // load Asub
        const unsigned int A_col = BLOCK_SIZE * i + col;
        if (grid_row < M && A_col < K) {
            As[row][col] = A[M * A_col + grid_row];
        }
        // load Bsub
        const unsigned int B_row = row + BLOCK_SIZE * i;
        if (B_row < K && grid_col < N) {
            Bs[row][col] = B[K * grid_col + B_row];
        }

        __syncthreads();

        unsigned int num_elems = BLOCK_SIZE;
        if ((K - i * BLOCK_SIZE) < BLOCK_SIZE) {
            num_elems = K - i * BLOCK_SIZE;
        }
        for (int j = 0; j < num_elems; ++j) {
            value += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    if (grid_row < M && grid_col < N) {
        const unsigned int index = M * grid_col + grid_row;
        value = alpha * value + beta * C[index];
        C[index] = 1.0 / (1.0 + std::exp(-value));
    }
}

void GEMMSigmoid(double* A, double* B, double* C, const double alpha, const double beta,
                 const int M, const int N, const int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    gpu_GEMMSigmoid<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

__global__
void gpu_GEMMT1(double* A, double* B, double* C, const double alpha, const double beta,
                     const int M, const int N, const int K) {
    // thread row and column within Csub
    const unsigned int row = threadIdx.x;
    const unsigned int col = threadIdx.y;
    // index within grid
    const unsigned int grid_row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_col = blockIdx.y * blockDim.y + threadIdx.y;

    double value = 0;
    const unsigned int iter = ceil(K / float(BLOCK_SIZE));
    for (int i = 0; i < iter; ++i) {
        // shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];

        // load Asub
        const unsigned int A_col = BLOCK_SIZE * i + col;
        if (grid_row < M && A_col < K) {
            As[row][col] = A[K * grid_row + A_col];
        }
        // load Bsub
        const unsigned int B_row = row + BLOCK_SIZE * i;
        if (B_row < K && grid_col < N) {
            Bs[row][col] = B[K * grid_col + B_row];
        }

        __syncthreads();

        unsigned int num_elems = BLOCK_SIZE;
        if ((K - i * BLOCK_SIZE) < BLOCK_SIZE) {
            num_elems = K - i * BLOCK_SIZE;
        }
        for (int j = 0; j < num_elems; ++j) {
            value += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    if (grid_row < M && grid_col < N) {
        const unsigned int index = M * grid_col + grid_row;
        C[index] = alpha * value + beta * C[index];
    }
}

void GEMMT1(double* A, double* B, double* C, const double alpha, const double beta,
            const int M, const int N, const int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_GEMMT1<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

__global__
void gpu_GEMMT2(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                const double alpha, const double beta, const int M, const int N, const int K) {
    // thread row and column within Csub
    const unsigned int row = threadIdx.x;
    const unsigned int col = threadIdx.y;
    // index within grid
    const unsigned int grid_row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_col = blockIdx.y * blockDim.y + threadIdx.y;

    double value = 0.0;
    const unsigned int iter = ceil(K / float(BLOCK_SIZE));
    for (int i = 0; i < iter; ++i) {
        // shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];
        // load Asub
        const unsigned int A_col = BLOCK_SIZE * i + col;
        if (grid_row < M && A_col < K) {
            As[row][col] = A[M * A_col + grid_row];
        }
        // load Bsub (transpose)
        const unsigned int B_row = row + BLOCK_SIZE * i;
        if (B_row < K && grid_col < N) {
            Bs[row][col] = B[N * B_row + grid_col];
        }

        __syncthreads();

        unsigned int num_elems = BLOCK_SIZE;
        if ((K - i * BLOCK_SIZE) < BLOCK_SIZE) {
            num_elems = K - i * BLOCK_SIZE;
        }
        for (int j = 0; j < num_elems; ++j) {
            value += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    if (grid_row < M && grid_col < N) {
        const unsigned int index = M * grid_col + grid_row;
        C[index] = alpha * value + beta * C[index];
    }
}

void GEMMT2(double* A, double* B, double* C, const double alpha, const double beta,
            const int M, const int N, const int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_GEMMT2<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}

/*
 * Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
 * A: (M, K), B: (K, N), C: (M, N)
 * All matrices are organized in column-major order.
 */

/* Algorithm 2:
 * Shared memory
 */
__global__
void gpu_GEMM(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
              const double alpha, const double beta, const int M, const int N, const int K) {
    // thread row and column within Csub
    const unsigned int row = threadIdx.x;
    const unsigned int col = threadIdx.y;
    // index within grid
    const unsigned int grid_row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int grid_col = blockIdx.y * blockDim.y + threadIdx.y;

    double value = 0.0;
    const unsigned int iter = ceil(K / float(BLOCK_SIZE));
    for (int i = 0; i < iter; ++i) {
        // shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE+1];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE+1];
        // load Asub
        const unsigned int A_col = BLOCK_SIZE * i + col;
        if (grid_row < M && A_col < K) {
            As[row][col] = A[M * A_col + grid_row];
        }
        // load Bsub
        const unsigned int B_row = row + BLOCK_SIZE * i;
        if (B_row < K && grid_col < N) {
            Bs[row][col] = B[K * grid_col + B_row];
        }

        __syncthreads();

        unsigned int num_elems = BLOCK_SIZE;
        if ((K - i * BLOCK_SIZE) < BLOCK_SIZE) {
            num_elems = K - i * BLOCK_SIZE;
        }
        for (int j = 0; j < num_elems; ++j) {
            value += As[row][j] * Bs[j][col];
        }

        __syncthreads();
    }

    if (grid_row < M && grid_col < N) {
        const unsigned int index = M * grid_col + grid_row;
        C[index] = alpha * value + beta * C[index];
    }
}

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    gpu_GEMM<<<grid, block>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}

/* Algorithm 1:
 * Each thread computes one element of C by accumulating results into value
 */
__global__
void gpu_GEMM_1(double* A, double* B, double* C, double alpha, double beta, int M,
                  int N, int K) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        double value = 0.0;
        const unsigned int index = M * col + row;
        for (size_t i = 0; i < K; ++i) {
            value += alpha * A[M * i + row] * B[K * col + i];
        }
        value += beta * C[index];
        C[index] = value;
    }
}

int myGEMM_1(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_GEMM_1<<<grid, block>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}
