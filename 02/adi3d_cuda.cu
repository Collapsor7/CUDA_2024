#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

#define Max(a, b) fmax(a, b)
#define BLOCKSIZE 4


void init(double *a, int L);
void runOnCPU(double *a, int itmax, double maxeps, double *elapsedTime, int L);
void runOnGPU_GaussSeidel(double *a, int itmax, double maxeps, double *elapsedTime, int L);

__global__ void kernel_update_i(double *a_new, double *a_old, int L);
__global__ void kernel_update_j(double *a_new, double *a_old, int L);
__global__ void kernel_update_k(double *a_new, double *a_old, double *eps_d, int L);
//__global__ void kernel_update_all(double *a_new, double *a_old, double *eps_d, int L);
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

    int L = (int)pow((free_mem * 0.8 / (3 * sizeof(double))), 1.0 / 3.0); 
    printf("Dynamic grid size set to: %d x %d x %d\n", L, L, L);
    L = 332; 

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
        runOnGPU_GaussSeidel(a, itmax, maxeps, &elapsedTime, L);
    }

    printf("ADI Benchmark Completed.\n");
    printf("Size            = %4d x %4d x %4d\n", L, L, L);
    printf("Iterations      =       %12d\n", itmax);
    printf("Time in seconds =       %12.6lf\n", elapsedTime);
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
                    a[i * L * L + j * L + k] = 0.0;
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

__global__ void kernel_update_i(double *a_new, double *a_old, int L)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i_layer = 1 ; i_layer < L - 1 ; i_layer++)
        if (j > 0 && j < L - 1 && k > 0 && k < L - 1)
        {
            a_new[i_layer * L * L + j * L + k] = (a_new[(i_layer - 1) * L * L + j * L + k] + a_old[(i_layer + 1) * L * L + j * L + k]) / 2.0;
            // printf("a[%d,%d,%d]=%.6lf\n",i_layer-1,j,k,a_new[(i_layer - 1) * L * L + j * L + k]);
            // printf("a[%d,%d,%d]=%.6lf\n",i_layer+1,j,k,a_old[(i_layer + 1) * L * L + j * L + k]);
            // printf("kernel_update_i: i=%d, j=%d, k=%d, a_new=%.6lf\n", i_layer, j, k, a_new[i_layer * L * L + j * L + k]);
        }
}

__global__ void kernel_update_j(double *a_new, double *a_old, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    for (int j_layer= 1 ; j_layer < L - 1 ; j_layer++)
        if (i > 0 && i < L - 1 && k > 0 && k < L - 1)
        {
            a_new[i * L * L + j_layer * L + k] = (a_new[i * L * L + (j_layer-1) * L + k] + a_old[i * L * L + (j_layer+1) * L + k]) / 2.0;
            // printf("a[%d,%d,%d]=%.6lf\n",i, j_layer-1,k,a_new[i * L * L + (j_layer-1) * L + k]);
            // printf("a[%d,%d,%d]=%.6lf\n",i, j_layer+1,k,a_old[i * L * L + (j_layer+1) * L + k]);
            // printf("kernel_update_j: i=%d, j=%d, k=%d, a_new=%.6lf\n", i, j_layer, k, a_new[i * L * L + j_layer * L + k]);
        }
}

__global__ void kernel_update_k(double *a_new, double *a_old, double *eps_d, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ double shared_eps[];

    double local_eps = 0.0;

    for (int k_layer = 1 ; k_layer < L - 1 ; k_layer++)
        if (i> 0 && i < L - 1 && j > 0 && j < L - 1)
        {
            double tmp = (a_new[i * L * L + j * L + (k_layer - 1)] + a_old[i * L * L + j * L + (k_layer + 1)]) / 2.0;
            double diff = fabs(a_old[i * L * L + j * L + k_layer] - tmp);
            a_new[i * L * L + j * L + k_layer] = tmp;
            local_eps = Max(diff, local_eps);
            // printf("a[%d,%d,%d]=%.6lf\n",i,j,k_layer-1,a_new[i * L * L + j * L + (k_layer-1)]);
            // printf("a[%d,%d,%d]=%.6lf\n",i,j,k_layer+1,a_old[i * L * L + j * L + (k_layer+1)]);
            // printf("kernel_update_k: i=%d, j=%d, k=%d, a_new=%.6lf, diff=%.6lf\n", i, j, k_layer, tmp, diff);
        }

    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    if (i < L && j < L)
        shared_eps[threadId] = local_eps;
    else
        shared_eps[threadId] = 0.0;
    __syncthreads();

    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1)
    {
        if (threadId < stride)
        {
            shared_eps[threadId] = Max(shared_eps[threadId], shared_eps[threadId + stride]);
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        atomicMaxDouble(eps_d, shared_eps[0]);
    }
}

void runOnGPU_GaussSeidel(double *a, int itmax, double maxeps, double *elapsedTime, int L)
{
    double *a_old, *a_new, *eps_d, eps;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&a_old, L * L * L * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc a_old failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    cudaStatus = cudaMalloc(&a_new, L * L * L * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc a_new failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(a_old);
        return;
    }

    cudaStatus = cudaMalloc(&eps_d, sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc eps_d failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(a_old);
        cudaFree(a_new);
        return;
    }

    cudaStatus = cudaMemcpy(a_old, a, L * L * L * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy to a_old failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(a_old);
        cudaFree(a_new);
        cudaFree(eps_d);
        return;
    }


    cudaStatus = cudaMemcpy(a_new, a_old, L * L * L * sizeof(double), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy to a_new failed! Error: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(a_old);
        cudaFree(a_new);
        cudaFree(eps_d);
        return;
    }


    dim3 threads(BLOCKSIZE, BLOCKSIZE, 1); 
    dim3 blocks((L + threads.x - 1) / threads.x, 
                (L + threads.y - 1) / threads.y, 
                1);



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 1; it <= itmax; it++)
    {

        cudaStatus = cudaMemset(eps_d, 0, sizeof(double));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemset eps_d failed! Error: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        //kernel_update_all<<<blocks, threads>>>(a_new,a_old,eps_d,L);
 
        kernel_update_i<<<blocks, threads>>>(a_new, a_old, L);
            

        kernel_update_j<<<blocks, threads>>>(a_old, a_new, L);

        size_t sharedMemSize = threads.x * threads.y * sizeof(double);
        kernel_update_k<<<blocks, threads, sharedMemSize>>>(a_new, a_old, eps_d, L);

        cudaStatus = cudaDeviceSynchronize();
        cudaStatus = cudaMemcpy(&eps, eps_d, sizeof(double), cudaMemcpyDeviceToHost);


        printf("GPU IT = %4d   EPS = %14.7E \n", it, eps);
        if (eps < maxeps) break;


        double *temp = a_old;
        a_old = a_new;
        a_new = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);
    *elapsedTime = (double)(gpuTime) / 1000.0; 

    cudaFree(a_old);
    cudaFree(a_new);
    cudaFree(eps_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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
                {
                    a[i * L * L + j * L + k] = (a[(i - 1) * L * L + j * L + k] + a[(i + 1) * L * L + j * L + k]) / 2.0;
                    printf("a[%d,%d,%d]=%.6lf \n",i-1,j,k,a[(i-1) * L * L + j * L + k]);
                    printf("a[%d,%d,%d]=%.6lf \n",i+1,j,k,a[(i+1) * L * L + j * L + k]);
                    printf("current_i_thread, i = %d, j = %d, k = %d, a = %.6f.\n",i,j,k,a[i * L * L + j * L + k]);
                }

        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                {
                    a[i * L * L + j * L + k] = (a[i * L * L + (j - 1) * L + k] + a[i * L * L + (j + 1) * L + k]) / 2.0;
                    printf("a[%d,%d,%d]=%.6lf \n",i,j-1,k,a[i * L * L + (j-1) * L + k]);
                    printf("a[%d,%d,%d]=%.6lf \n",i,j+1,k,a[i * L * L + (j+1) * L + k]);
                    printf("current_j_thread, i = %d, j = %d, k = %d, a = %.6f.\n",i,j,k,a[i * L * L + j * L + k]);
                }

        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                {
                    double tmp = (a[i * L * L + j * L + (k - 1)] + a[i * L * L + j * L + (k + 1)]) / 2.0;
                    eps = Max(eps, fabs(a[i * L * L + j * L + k] - tmp));
                    a[i * L * L + j * L + k] = tmp;
                    printf("a[%d,%d,%d]=%.6lf \n",i,j,k-1,a[i * L * L + j * L + (k-1)]);
                    printf("a[%d,%d,%d]=%.6lf \n",i,j,k+1,a[i * L * L + j * L + (k+1)]);
                    printf("current_k_thread, i = %d, j = %d, k = %d, a = %.6f, diff = %.6f.\n",i,j,k,a[i * L * L + j * L + k],eps);
                }

        printf(" CPU IT = %4d   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }
    clock_t end = clock();
    *elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
}
