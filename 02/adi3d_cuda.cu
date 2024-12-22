#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <string.h>

#define Max(a, b) fmax(a, b)
#define INDEX(i,j,k) ((i) * L * L + (j) * L + (k))
#define BLOCKSIZE 4
#define DIMS 32
#define ROWS 8


void init(double *a, int L);
void runOnCPU(double *a, int itmax, double maxeps, double *elapsedTime, int L);
void runOnGPU(double *a, int itmax, double maxeps, double *elapsedTime, int L);

__global__ void kernel_update_i(double *a, int L, double *eps_d);
__global__ void kernel_update_j(double *a, int L, double *eps_d);
__global__ void transposeMatrix(const double* in, double* out, int L);
__global__ void kernel_compute_eps(const double *a_old, const double *a_new, double *eps_d, int L);
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

    int L = (int)pow((free_mem * 0.8 / (2 *sizeof(double))), 1.0 / 3.0); 
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
        runOnGPU(a, itmax, maxeps, &elapsedTime, L);
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

__global__ void kernel_update_i(double *a, int L)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

   
    if (j > 0 && j < L - 1)
    { 
        if (k > 0 && k < L - 1){
            for (int i_layer = 1 ; i_layer < L - 1 ; i_layer++)
            {
                a[INDEX(i_layer,j,k)] = (a[INDEX(i_layer-1,j,k)] + a[INDEX(i_layer+1,j,k)]) / 2.0;
                // printf("a[%d,%d,%d]=%.6lf\n",i_layer-1,j,k,a[(i_layer - 1) * L * L + j * L + k]);
                // //printf("a[%d,%d,%d]=%.6lf\n",i_layer,j,k-1,a[INDEX(i_layer,j,k-1)]);
                // printf("a[%d,%d,%d]=%.6lf\n",i_layer+1,j,k,a[(i_layer + 1) * L * L + j * L + k]);
                // printf("kernel_update_i: i=%d, j=%d, k=%d, a_new=%.6lf\n", i_layer, j, k, a[i_layer * L * L + j * L + k]);
            }  
        }

    }
}

__global__ void kernel_update_j(double *a, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (i > 0 && i < L - 1 && k > 0 && k < L - 1)
    {

        for (int j_layer= 1 ; j_layer < L - 1 ; j_layer++){
            a[INDEX(i,j_layer,k)] = (a[INDEX(i,j_layer-1,k)] + a[INDEX(i,j_layer+1,k)]) / 2.0;
            // printf("a[%d,%d,%d]=%.6lf\n",i, j_layer-1,k,a[i * L * L + (j_layer-1) * L + k]);
            // printf("a[%d,%d,%d]=%.6lf\n",i, j_layer+1,k,a[i * L * L + (j_layer+1) * L + k]);
            // printf("kernel_update_j: i=%d, j=%d, k=%d, a_new=%.6lf\n", i, j_layer, k, a[i * L * L + j_layer * L + k]);
        }
    }
 
               
}

// __global__ void kernel_update_k(double *a_new, double *a_old, double *eps_d, int L)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     extern __shared__ double shared_eps[];

//     double local_eps = 0.0;


//     if (i> 0 && i < L - 1)
//     {
//         if(j > 0 && j < L - 1)
//             for (int k_layer = 1 ; k_layer < L - 1 ; k_layer++)
//             {
//                 double tmp = (a_new[INDEX(i,j,k_layer-1)] + a_old[INDEX(i,j,k_layer+1)]) / 2.0;
//                 double diff = fabs(a_old[INDEX(i,j,k_layer)] - tmp);
//                 a_new[INDEX(i,j,k_layer)] = tmp;
//                 local_eps = Max(diff, local_eps);
//                 // printf("a[%d,%d,%d]=%.6lf\n",i,j,k_layer-1,a_new[i * L * L + j * L + (k_layer-1)]);
//                 // printf("a[%d,%d,%d]=%.6lf\n",i,j,k_layer+1,a_old[i * L * L + j * L + (k_layer+1)]);
//                 // printf("kernel_update_k: i=%d, j=%d, k=%d, a_new=%.6lf, diff=%.6lf\n", i, j, k_layer, tmp, diff);
//             }
//     }

//     int threadId = threadIdx.x + threadIdx.y * blockDim.x;
//     if (i < L && j < L)
//         shared_eps[threadId] = local_eps;
//     else
//         shared_eps[threadId] = 0.0;
//     __syncthreads();

//     for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1)
//     {
//         if (threadId < stride)
//         {
//             shared_eps[threadId] = Max(shared_eps[threadId], shared_eps[threadId + stride]);
//         }
//         __syncthreads();
//     }

//     if (threadId == 0)
//     {
//         atomicMaxDouble(eps_d, shared_eps[0]);
//     }
// }


//(i,j,k)->(k,j,i)
__global__ void transposeMatrix(const double* in, double* out, int L) 
{
    int j = blockIdx.y; 
    __shared__ double storey[DIMS + 1][ DIMS + 1]; 

    int i = blockIdx.x * DIMS + threadIdx.y; 
    int k = blockIdx.z * DIMS + threadIdx.x; 

    if (k >= 0 && k < L) { 
        for (int dim = 0; dim < DIMS; dim += ROWS) 
        { 
            int row = dim + i; 
            if (row >= L) 
                break; 
            storey[threadIdx.y + dim][threadIdx.x] = in[INDEX(i + dim,j,k)]; 
        } 
    } 
    __syncthreads(); 

    int kk = blockIdx.x * DIMS + threadIdx.x; 
    int ii = blockIdx.z * DIMS + threadIdx.y; 
    if (kk >= 0 && kk < L) { 
        for (int dim = 0; dim < DIMS; dim += ROWS) 
        {   
            int row = dim + ii; 
            if (row >= L) 
                break; 
            out[INDEX(ii + dim, j, kk)] = storey[threadIdx.x][threadIdx.y + dim]; 
        } 
    } 
    //__syncthreads();
}


void runOnGPU(double *a, int itmax, double maxeps, double *elapsedTime, int L) {

    double *a_old, *a_new, *transposed, *eps_d, eps;

    cudaMalloc(&a_old, L * L * L * sizeof(double));
    cudaMalloc(&a_new, L * L * L * sizeof(double));
    cudaMalloc(&transposed, L * L * L * sizeof(double)); 
    cudaMalloc(&eps_d, sizeof(double));

    cudaMemcpy(a_old, a, L * L * L * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(a_new, a_old, L * L * L * sizeof(double), cudaMemcpyDeviceToDevice);

    dim3 threads(BLOCKSIZE, BLOCKSIZE,1);
    dim3 blocks((L + BLOCKSIZE - 1) / BLOCKSIZE, (L + BLOCKSIZE - 1) / BLOCKSIZE,1);
  
    dim3 threads_trans(DIMS,ROWS);
    dim3 grid_trans((L + DIMS - 1) / DIMS, L, (L + DIMS - 1) / DIMS);
    dim3 grid_deal_with_i_trans((L + BLOCKSIZE - 1) / BLOCKSIZE, (L + BLOCKSIZE - 1) / BLOCKSIZE);

    dim3 threads_eps(BLOCKSIZE, BLOCKSIZE,BLOCKSIZE);
    dim3 blocks_eps((L + BLOCKSIZE - 1) / BLOCKSIZE, 
                (L + BLOCKSIZE - 1) / BLOCKSIZE, 
                (L + BLOCKSIZE - 1) / BLOCKSIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 1; it <= itmax; it++) {
        cudaMemset(eps_d, 0, sizeof(double));

        kernel_update_i<<<blocks, threads>>>(a_old, L);
        cudaDeviceSynchronize();

        kernel_update_j<<<blocks, threads>>>(a_old, L);

        cudaDeviceSynchronize();

        transposeMatrix<<<grid_trans, threads_trans>>>(a_old, transposed, L);
        cudaDeviceSynchronize();

        kernel_update_i<<<grid_deal_with_i_trans, threads>>>(transposed, L);
        cudaDeviceSynchronize();

        transposeMatrix<<<grid_trans, threads_trans>>>(transposed,a_new, L);
        cudaDeviceSynchronize();

        // kernel_compute_eps<<<blocks_eps, threads_eps>>>(a_old, a_new, eps_d, L);
        size_t shared_mem_size = BLOCKSIZE * BLOCKSIZE * BLOCKSIZE * sizeof(double); 
        kernel_compute_eps<<<blocks_eps, threads_eps, shared_mem_size>>>(a_old, a_new, eps_d, L);
        cudaDeviceSynchronize();

        cudaMemcpy(&eps, eps_d, sizeof(double), cudaMemcpyDeviceToHost);

        printf("GPU IT = %4d   EPS = %14.7E\n", it, eps);
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
    cudaFree(transposed);
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
                    // printf("a[%d,%d,%d]=%.6lf \n",i-1,j,k,a[(i-1) * L * L + j * L + k]);
                    // printf("a[%d,%d,%d]=%.6lf \n",i+1,j,k,a[(i+1) * L * L + j * L + k]);
                    // printf("current_i_thread, i = %d, j = %d, k = %d, a = %.6f.\n",i,j,k,a[i * L * L + j * L + k]);
                }

        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                {
                    a[i * L * L + j * L + k] = (a[i * L * L + (j - 1) * L + k] + a[i * L * L + (j + 1) * L + k]) / 2.0;
                    // printf("a[%d,%d,%d]=%.6lf \n",i,j-1,k,a[i * L * L + (j-1) * L + k]);
                    // printf("a[%d,%d,%d]=%.6lf \n",i,j+1,k,a[i * L * L + (j+1) * L + k]);
                    // printf("current_j_thread, i = %d, j = %d, k = %d, a = %.6f.\n",i,j,k,a[i * L * L + j * L + k]);
                }

        for (int i = 1; i < L - 1; i++)
            for (int j = 1; j < L - 1; j++)
                for (int k = 1; k < L - 1; k++)
                {
                    double tmp = (a[i * L * L + j * L + (k - 1)] + a[i * L * L + j * L + (k + 1)]) / 2.0;
                    eps = Max(eps, fabs(a[i * L * L + j * L + k] - tmp));
                    a[i * L * L + j * L + k] = tmp;
                    // printf("a[%d,%d,%d]=%.6lf \n",i,j,k-1,a[i * L * L + j * L + (k-1)]);
                    // printf("a[%d,%d,%d]=%.6lf \n",i,j,k+1,a[i * L * L + j * L + (k+1)]);
                    // printf("current_k_thread, i = %d, j = %d, k = %d, a = %.6f, diff = %.6f.\n",i,j,k,a[i * L * L + j * L + k],eps);
                }

        printf(" CPU IT = %4d   EPS = %14.7E\n", it, eps);
        if (eps < maxeps)
            break;
    }
    clock_t end = clock();
    *elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
}

__global__ void kernel_compute_eps(const double *a_old, const double *a_new, double *eps_d, int L)
{
    extern __shared__ double sdata[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    double diff = 0.0;
    if (i < L && j < L && k < L)
    {
        
        diff = fabs(a_new[INDEX(i,j,k)] - a_old[INDEX(i,j,k)]);
        //printf("diff_a[%d,%d,%d]=%.6lf \n",i,j,k,diff);
    }

   
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    sdata[tid] = diff;
    __syncthreads();

    for (unsigned int s = blockDim.x * blockDim.y * blockDim.z / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = Max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicMaxDouble(eps_d, sdata[0]);
    }
}