#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define L 384
#define ITMAX 100
#define MAXEPS 0.5f

#define cudaCheckError() {                                         \
    cudaError_t e = cudaGetLastError();                            \
    if (e != cudaSuccess) {                                        \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
               cudaGetErrorString(e));                             \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
}

__device__ inline double Max(double a, double b) {
    return a > b ? a : b;
}

// Kernel to update matrix A
__global__ void jacobi_update_A(double *A, double *B, unsigned long long int *eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < L - 1 && j < L - 1 && k < L - 1) {
        int idx = i * L * L + j * L + k;
        double tmp = fabs(B[idx] - A[idx]);
        unsigned long long int tmp_int = __double_as_longlong(tmp);
        atomicMax(eps, tmp_int); // Atomic operation to update eps
        A[idx] = B[idx];
    }
}

// Kernel to update matrix B
__global__ void jacobi_update_B(double *A, double *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i < L - 1 && j < L - 1 && k < L - 1) {
        int idx = i * L * L + j * L + k;
        B[idx] = (A[(i - 1) * L * L + j * L + k] +
                  A[i * L * L + (j - 1) * L + k] +
                  A[i * L * L + j * L + (k - 1)] +
                  A[i * L * L + j * L + (k + 1)] +
                  A[i * L * L + (j + 1) * L + k] +
                  A[(i + 1) * L * L + j * L + k]) / 6.0f;
    }
}

int main() {
    double *A, *B;
    double eps;
    unsigned long long int *d_eps;
    double *d_A, *d_B;

    size_t size = L * L * L * sizeof(double);
    A = (double *)malloc(size);
    B = (double *)malloc(size);

    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < L; k++) {
                int idx = i * L * L + j * L + k;
                A[idx] = 0;
                if (i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1)
                    B[idx] = 0;
                else
                    B[idx] = 4 + i + j + k;
            }

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_eps, sizeof(unsigned long long int));
    cudaCheckError();

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 block(8, 8, 8);
    dim3 grid((L + block.x - 2) / block.x, (L + block.y - 2) / block.y, (L + block.z - 2) / block.z);

    for (int it = 1; it <= ITMAX; it++) {
        eps = 0;
        cudaMemcpy(d_eps, &eps, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

        jacobi_update_A<<<grid, block>>>(d_A, d_B, d_eps);
        cudaCheckError();

        jacobi_update_B<<<grid, block>>>(d_A, d_B);
        cudaCheckError();

        cudaMemcpy(&eps, d_eps, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        eps = *reinterpret_cast<double*>(&eps);

        printf(" IT = %4d   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }

    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_eps);

    printf(" Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", ITMAX);
    //TODO
    //printf(" Time in seconds =       %12.2lf\n", endt - startt);
    printf(" Operation type  =     floating point\n");
    //printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));

    printf(" END OF Jacobi3D Benchmark\n");


    return 0;
}