#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

#define Max(a, b) fmax(a,b)
#define BLOCKSIZE 4

void init(double *a, int L);
void runOnCPU(double *a, int itmax, double maxeps, double *elapsedTime, int L);
void runOnGPU(double *a, int itmax, double maxeps, double *elapsedTime, int L);
void runOnGPU_double_buffering(double *a, int itmax, double maxeps, double *elapsedTime, int L);

__global__ void kernel_update(double *a, double *eps_d, int L);
__global__ void kernel_update_i(double *a, int L);
__global__ void kernel_update_j(double *a, int L);
__global__ void kernel_update_k(double *a, double *eps_d, int L);
__global__ void kernel_update_all(double *a_old, double *a_new, double *eps_d, int L);

__device__ double atomicMaxDouble(double* addr, double value);

int main(int argc, char *argv[])
{
    double maxeps = 0.01; 
    int itmax = 100;
    double *a;
    double elapsedTime = 0.0;

    if (argc != 2)
    {
        printf("Usage: ./cuda2 <mode>\n");
        printf("Where <mode> is either 'cpu' or 'gpu'\n");
        return -1;
    }

    if (strcmp(argv[1], "cpu") == 0)
    {
        printf("Running on CPU...\n");
    }
    else if (strcmp(argv[1], "gpu") == 0)
    {
        printf("Running on GPU...\n");
    }
    else
    {
        printf("Invalid mode. Usage: ./cuda2 <mode> (mode: cpu or gpu)\n");
        return -1;
    }

    size_t free_mem, total_mem;
    cudaError_t cudaStatus = cudaMemGetInfo(&free_mem, &total_mem);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemGetInfo failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    printf("Free memory: %zu bytes\n", free_mem);
    printf("Total memory: %zu bytes\n", total_mem);


    int L = (int)pow((free_mem * 0.8 / (2 * sizeof(double))), 1.0 / 3.0); 
    printf("Dynamic grid size set to: %d x %d x %d\n", L, L, L);


    cudaStatus = cudaMallocHost((void **)&a, L * L * L * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    init(a, L);


    if (strcmp(argv[1], "cpu") == 0)
    {
        runOnCPU(a, itmax, maxeps, &elapsedTime, L);
    }
    else if (strcmp(argv[1], "gpu") == 0)
    {
        runOnGPU(a, itmax, maxeps, &elapsedTime, L);
        //runOnGPU_double_buffering(a, itmax, maxeps, &elapsedTime, L);
    }

    printf("ADI Benchmark Completed.\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      =       %12d\n", itmax);
    printf("Time in seconds =       %12.2lf\n", elapsedTime);
    printf("Operation type  =   double precision\n");
    printf("END OF ADI Benchmark\n");

    cudaFreeHost(a);
    return 0;
}

void init(double *a, int L)
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

__device__ double atomicMaxDouble(double* addr, double value)
{
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old, assumed;
    double old_val, new_val;

    while (true) {
        old = *addr_as_ull;
        old_val = __longlong_as_double(old);
        new_val = Max(value, old_val);
        unsigned long long int new_ull = __double_as_longlong(new_val);
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed, new_ull);
        if (assumed == old) {
            break;
        }
    }
    return __longlong_as_double(old);
}

__global__ void kernel_update_i(double *a, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1)
    {
        a[i * L * L + j * L + k] = (a[(i - 1) * L * L + j * L + k] + a[(i + 1) * L * L + j * L + k]) / 2.0;
    }
    __syncthreads();
}

__global__ void kernel_update_j(double *a, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1)
    {
        a[i * L * L + j * L + k] = (a[i * L * L + (j - 1) * L + k] + a[i * L * L + (j + 1) * L + k]) / 2.0;
    }
    __syncthreads();
}

__global__ void kernel_update_k(double *a, double *eps_d, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    double local_eps = 0.0;

    __shared__ double shared_eps[BLOCKSIZE*BLOCKSIZE*BLOCKSIZE]; 
    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    shared_eps[threadId] = 0.0;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1)
    {
        double tmp = (a[i * L * L + j * L + (k - 1)] + a[i * L * L + j * L + (k + 1)]) / 2.0;
        double diff = fabs(a[i * L * L + j * L + k] - tmp);
        a[i * L * L + j * L + k] = tmp;
        //atomicMax((unsigned long long int *)eps_d,__double_as_longlong(local_eps));
        //printf("GPU_temp : %f \n",tmp);
        //atomicMaxDouble(eps_d, max_eps_in_block);
        local_eps = Max(diff,local_eps);
      
    }
    __syncthreads(); 

    shared_eps[threadId] = local_eps;

    __syncthreads(); 

    if (threadId == 0){
        double max_eps_in_block = local_eps;
        for (int idx = 0; idx < blockDim.x * blockDim.y * blockDim.z; ++idx)
        {
            max_eps_in_block = Max(max_eps_in_block, shared_eps[idx]);
        }
        //atomicMaxDouble(eps_d, max_eps_in_block);
        atomicMax((unsigned long long int *)eps_d,__double_as_longlong(max_eps_in_block));
    }
        
        //__syncthreads(); 
        //printf(max_eps_in_block);
        //atomicMax((unsigned long long int *)eps_d,__double_as_longlong(max_eps_in_block));        
}

void runOnCPU(double *a, int itmax, double maxeps, double *elapsedTime, int L)
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
                    //printf("CPU_temp : %f \n",tmp);
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
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&a_d, L * L * L * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc a_d failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    cudaStatus = cudaMalloc(&eps_d, sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc eps_d failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(a_d);
        return;
    }

    cudaStatus = cudaMemcpy(a_d, a, L * L * L * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy to a_d failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(a_d);
        cudaFree(eps_d);
        return;
    }

    dim3 threads(BLOCKSIZE,BLOCKSIZE,BLOCKSIZE);
    dim3 blocks((L + threads.x - 1) / threads.x, 
                (L + threads.y - 1) / threads.y, 
                (L + threads.z - 1) / threads.z);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 1; it <= itmax; it++)
    {
 
        cudaStatus = cudaMemset(eps_d, 0, sizeof(double));

        kernel_update_i<<<blocks, threads>>>(a_d, L);
        cudaStatus = cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();


        kernel_update_j<<<blocks, threads>>>(a_d, L);
        cudaStatus = cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();


        kernel_update_k<<<blocks, threads>>>(a_d, eps_d, L);
        //kernel_update<<<blocks, threads>>>(a_d,eps_d, L);
        cudaStatus = cudaDeviceSynchronize();
        cudaStatus = cudaGetLastError();
 
        cudaStatus = cudaMemcpy(&eps, eps_d, sizeof(double), cudaMemcpyDeviceToHost);
        cudaStatus = cudaDeviceSynchronize();

        printf("GPU IT = %4d   EPS = %14.7E \n", it, eps);
        if (eps < maxeps) break;
        
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);
    *elapsedTime = (double)(gpuTime) / 1000.0; 

    cudaFree(a_d);
    cudaFree(eps_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

