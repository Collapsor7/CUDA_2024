#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

#define MAXEPS 0.5f
#define ITMAX 100
#define BLOCK_SIZE 8

#define cudaCheckError() {                                         \
    cudaError_t e = cudaGetLastError();                            \
    if (e != cudaSuccess) {                                        \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
               cudaGetErrorString(e));                             \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

// Host and Device utility function
__device__ __host__ inline double Max(double a, double b) {
    return a > b ? a : b;
}

// GPU Kernel: Update A matrix
__global__ void jacobi_update_A(double *A, double *B, double *eps, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < L - 1 && j < L - 1 && k < L - 1) {
        int idx = i * L * L + j * L + k;
        if (idx < L * L * L) { // Ensure index is within bounds
            double diff = fabs(B[idx] - A[idx]);
            atomicMax((unsigned long long int *)eps, __double_as_longlong(diff)); // Use atomic for reduction
            A[idx] = B[idx];
        }
    }
}

// GPU Kernel: Update B matrix
__global__ void jacobi_update_B(double *A, double *B, int L) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < L - 1 && j < L - 1 && k < L - 1) {
        int idx = i * L * L + j * L + k;
        if (idx < L * L * L) { // Ensure index is within bounds
            B[idx] = (A[(i - 1) * L * L + j * L + k] +
                      A[i * L * L + (j - 1) * L + k] +
                      A[i * L * L + j * L + (k - 1)] +
                      A[i * L * L + j * L + (k + 1)] +
                      A[i * L * L + (j + 1) * L + k] +
                      A[(i + 1) * L * L + j * L + k]) / 6.0;
        }
    }
}

// CPU version of Jacobi update
void jacobi_cpu(double *A, double *B, double &eps, int L) {
    eps = 0.0;
    for (int i = 1; i < L - 1; ++i) {
        for (int j = 1; j < L - 1; ++j) {
            for (int k = 1; k < L - 1; ++k) {
                int idx = i * L * L + j * L + k;
                double diff = fabs(B[idx] - A[idx]);
                eps = Max(eps, diff);
                A[idx] = B[idx];
            }
        }
    }

    for (int i = 1; i < L - 1; ++i) {
        for (int j = 1; j < L - 1; ++j) {
            for (int k = 1; k < L - 1; ++k) {
                int idx = i * L * L + j * L + k;
                B[idx] = (A[(i - 1) * L * L + j * L + k] +
                          A[i * L * L + (j - 1) * L + k] +
                          A[i * L * L + j * L + (k - 1)] +
                          A[i * L * L + j * L + (k + 1)] +
                          A[i * L * L + (j + 1) * L + k] +
                          A[(i + 1) * L * L + j * L + k]) / 6.0;
            }
        }
    }
}

// Initialize the matrices
void initialize(double *A, double *B, int L) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                int idx = i * L * L + j * L + k;
                A[idx] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[idx] = 0;
                else
                    B[idx] = 4 + i + j + k;
            }
        }
    }
}

// Main program
int main(int argc, char **argv) {
    bool use_gpu = (argc > 1 && strcmp(argv[1], "gpu") == 0);
    int L = 384;

    if (use_gpu) {
        // GPU Execution
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        L = pow((free_mem * 0.9 / (2.0 * sizeof(double))), 1.0 / 3.0);

        printf("Running on GPU with L = %d\n", L);

        double *A, *B, *eps;
        cudaMallocManaged(&A, L * L * L * sizeof(double));
        cudaMallocManaged(&B, L * L * L * sizeof(double));
        cudaMallocManaged(&eps, sizeof(double));

        initialize(A, B, L);

        dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((L - 2 + block.x - 1) / block.x, (L - 2 + block.y - 1) / block.y, (L - 2 + block.z - 1) / block.z);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int it = 1; it <= ITMAX; it++) {
            *eps = 0.0;

            jacobi_update_A<<<grid, block>>>(A, B, eps, L);
            cudaDeviceSynchronize();
            cudaCheckError();

            jacobi_update_B<<<grid, block>>>(A, B, L);
            cudaDeviceSynchronize();
            cudaCheckError();

            printf(" IT = %4d   EPS = %14.7E\n", it, *eps);

            if (*eps < MAXEPS)
                break;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf(" GPU Time (ms): %f\n", milliseconds);

        cudaFree(A);
        cudaFree(B);
        cudaFree(eps);
    } else {
        // CPU Execution
        printf("Running on CPU with L = %d\n", L);

        double *A = (double *)malloc(L * L * L * sizeof(double));
        double *B = (double *)malloc(L * L * L * sizeof(double));
        double eps = 0.0;

        initialize(A, B, L);

        auto cpu_start = std::chrono::high_resolution_clock::now();

        for (int it = 1; it <= ITMAX; it++) {
            eps = 0.0;
            jacobi_cpu(A, B, eps, L);
            printf(" IT = %4d   EPS = %14.7E\n", it, eps);

            if (eps < MAXEPS)
                break;
        }

        auto cpu_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = cpu_stop - cpu_start;

        printf(" CPU Time (s): %f\n", elapsed.count());

        free(A);
        free(B);
    }

    printf(" Jacobi3D Benchmark Completed.\n");
    return 0;
}