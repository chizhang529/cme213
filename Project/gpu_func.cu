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
void gpu_softmax_kernel(double *mat, int M, int N) {
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

void gpu_softmax(double *mat, int M, int N) {
    dim3 block(BLOCK_SIZE);

    const unsigned int grid_x = ceil(N / (float)block.x);
    dim3 grid(grid_x);

    gpu_softmax_kernel<<<grid, block>>>(mat, M, N);
};

__global__
void gpu_sigmoid_kernel(double *mat, int M, int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        mat[index] = 1.0 / (1.0 + std::exp(-mat[index]));
    }
}

void gpu_sigmoid(double *mat, int M, int N) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_sigmoid_kernel<<<grid, block>>>(mat, M, N);
}


__global__
void gpu_transpose_kernel(double *mat1, double *mat2, int M, int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        const unsigned int new_index = N * row + col;
        mat2[new_index] = mat1[index];
    }
}

void gpu_transpose(double *mat1, double *mat2, int M, int N) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_transpose_kernel<<<grid, block>>>(mat1, mat2, M, N);
}

__global__
void gpu_linear_kernel(double *mat1, double *mat2, double alpha, double beta, int M, int N, int flag) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        if (!flag)
            mat2[index] = alpha * mat1[index] + beta * mat2[index];
        else
            mat2[index] = alpha * mat1[index] + beta * 1.0;
    }
}

void gpu_linear(double *mat1, double *mat2, double alpha, double beta, int M, int N, int flag) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_linear_kernel<<<grid, block>>>(mat1, mat2, alpha, beta, M, N, flag);
}

// __global__
// void gpu_sum_kernel(double *mat1, double *mat2, int M, int N, bool flag) {
//     const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

//     if (flag && index < M) {
//         double sum = 0.0;
//         for (size_t col = 0; col < N; ++col) {
//             sum += mat1[M * col + index];
//         }
//         mat2[index] = sum;
//     }

//     if (!flag && index < N) {
//         double sum = 0.0;
//         for (size_t row = 0; row < M; ++row) {
//             sum += mat1[M * index + row];
//         }
//         mat2[index] = sum;
//     }
// }

// void gpu_sum(double *mat1, double *mat2, int M, int N, int mode) {
//     // mode == 1: sum over row, 0: sum over column
//     dim3 block(BLOCK_SIZE);
//     dim3 grid(0);
//     if (mode) {
//         grid.x = ceil(N / (float)block.x);
//     } else {
//         grid.x = ceil(M / (float)block.x);
//     }
//     gpu_sum_kernel<<<grid, block>>>(mat1, mat2, M, N, mode);
// }

__global__
void gpu_row_sum_kernel(double *mat1, double *mat2, int M, int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        double sum = 0.0;
        for (size_t col = 0; col < N; ++col) {
            sum += mat1[M * col + row];
        }
        mat2[row] = sum;
    }
}

void gpu_row_sum(double *mat1, double *mat2, int M, int N) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(ceil(N / (float)block.x));
    gpu_row_sum_kernel<<<grid, block>>>(mat1, mat2, M, N);
}

__global__
void gpu_elem_mult_kernel(double* mat1, double* mat2, double alpha, int M, int N) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        const unsigned int index = M * col + row;
        mat2[index] *= alpha * mat1[index];
    }
}

void gpu_elem_mult(double* mat1, double* mat2, double alpha, int M, int N) {
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    gpu_elem_mult_kernel<<<grid, block>>>(mat1, mat2, alpha, M, N);
}

__global__
void myGEMMKernel(double* A, double* B, double* C, double alpha, double beta, int M,
                  int N, int K) {
    // each thread computes one element of C by accumulating results into value
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

/*
 * Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
 * A: (M, K), B: (K, N), C: (M, N)
 * All matrices are column-major.
 */
int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K) {
    /* Write an efficient GEMM implementation on GPU */
    dim3 block(BLOCK_SIZE, NUM_THREADS/BLOCK_SIZE);

    const unsigned int grid_x = ceil(M / (float)block.x);
    const unsigned int grid_y = ceil(N / (float)block.y);
    dim3 grid(grid_x, grid_y);

    myGEMMKernel<<<grid, block>>>(A, B, C, *alpha, *beta, M, N, K);
    return 0;
}
