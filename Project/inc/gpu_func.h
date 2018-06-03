#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaThreadSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

// helper functions
void gpu_sigmoid(double *mat, int M, int N);
void gpu_softmax(double *mat, int M, int N);
void gpu_transpose(double *mat1, double *mat2, int M, int N);
void gpu_linear(double *mat1, double *mat2, double alpha, double beta, int M, int N, int flag);
// void gpu_sum(double *mat1, double *mat2, int M, int N, int mode);
void gpu_row_sum(double *mat1, double *mat2, int M, int N);
void gpu_elem_mult(double* mat1, double* mat2, double alpha, int M, int N);

// general matrix multiplication
int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

#endif
