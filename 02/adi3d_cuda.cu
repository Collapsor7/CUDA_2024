#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))

void init(double *a,int L);
void runOnCPU(double *a, int itmax, double maxeps, double *elapsedTime,int L);
void runOnGPU(double *a, int itmax, double maxeps, double *elapsedTime,int L);

__global__ void kernelOptimized(double *a, double *eps_d, int L);

int main(int argc, char *argv[])
{
    double maxeps = 0.01;
    int itmax = 100;
    double *a;
    double elapsedTime = 0.0;

    // Allocate memory for 3D array
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Free memory: %zu bytes\n", free_mem);
    printf("Total memory: %zu bytes\n", total_mem);
    
    int L = (int)pow((free_mem * 0.9 / (2.0 * sizeof(double))), 1.0 / 3.0);

    printf("Dynamic grid size set to : %d x %d x %d \n",L,L,L);
    
    cudaMallocHost((void **)&a, L * L * L * sizeof(double));
    init(a,L);

    printf("Choose execution mode: 1 - CPU, 2 - GPU\n");
    int mode;
    scanf("%d", &mode);

    if (mode == 1)
    {
        printf("Running on CPU...\n");
        runOnCPU(a, itmax, maxeps, &elapsedTime, L);
    }
    else if (mode == 2)
    {
        printf("Running on GPU...\n");
        runOnGPU(a, itmax, maxeps, &elapsedTime, L);
    }
    else
    {
        printf("Invalid mode. Exiting...\n");
        cudaFreeHost(a);
        return 0;
    }

    // Print benchmark results
    printf(" ADI Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      =       %12d\n", itmax);
    printf(" Time in seconds =       %12.2lf\n", elapsedTime);
    printf(" Operation type  =   double precision\n");
    printf(" END OF ADI Benchmark\n");

    cudaFreeHost(a);
    return 0;
}

void init(double *a,int L)
{
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            for (int k = 0; k < L; k++)
            {
                if (k == 0 || k == L - 1 || j == 0 || j == L - 1 || i == 0 || i == L - 1)
                    a[i * L * L + j * L + k] = 10.0 * i / (L - 1) + 10.0 * j / (L - 1) + 10.0 * k / (L - 1);
                else
                    a[i * L * L + j * L + k] = 0;
            }
}

void runOnCPU(double *a, int itmax, double maxeps, double *elapsedTime,int L)
{
    double eps;
    clock_t start = clock();
    for (int it = 1; it <= itmax; it++)
    {
        eps = 0;
        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                    a[i * L * L + j * L + k] = (a[(i - 1) * L * L + j * L + k] + a[(i + 1) * L * L + j * L + k]) / 2;

        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                    a[i * L * L + j * L + k] = (a[i * L * L + (j - 1) * L + k] + a[i * L * L + (j + 1) * L + k]) / 2;

        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                {
                    double tmp = (a[i * L * L + j * L + (k - 1)] + a[i * L * L + j * L + (k + 1)]) / 2;
                    eps = Max(eps, fabs(a[i * L * L + j * L + k] - tmp));
                    a[i * L * L + j * L + k] = tmp;
                }

        printf(" CPU IT = %4d   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }
    clock_t end = clock();
    *elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
}

void runOnGPU(double *a, int itmax, double maxeps, double *elapsedTime, int L)
{
    double *a_d, *eps_d, eps;
    cudaMalloc((void **)&a_d, L * L * L * sizeof(double));
    cudaMemcpy(a_d, a, L * L * L * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&eps_d, sizeof(double));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((L + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (L + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (L + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int it = 1; it <= itmax; it++)
    {
        eps = 0;
        cudaMemcpy(eps_d, &eps, sizeof(double), cudaMemcpyHostToDevice);

        kernelOptimized<<<numBlocks, threadsPerBlock>>>(a_d, eps_d, L);
        cudaDeviceSynchronize();

        cudaMemcpy(&eps, eps_d, sizeof(double), cudaMemcpyDeviceToHost);
        printf(" GPU IT = %4d   EPS = %14.7E\n", it, eps);

        if (eps < maxeps)
            break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ElapsedTime;
    cudaMemcpy(a, a_d, L * L * L * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&ElapsedTime, start, stop);
    
    *elapsedTime = ElapsedTime / 1000.0;

    cudaFree(a_d);
    cudaFree(eps_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

__global__ void kernelOptimized(double *a, double *eps_d, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1)
    {
        double oldVal = a[i * L * L + j * L + k];
        double newVal = (a[(i - 1) * L * L + j * L + k] + a[(i + 1) * L * L + j * L + k]) / 2.0;
        atomicMax((unsigned long long *)eps_d, __double_as_longlong(fabs(oldVal - newVal)));
        a[i * L * L + j * L + k] = newVal;
    }
}

