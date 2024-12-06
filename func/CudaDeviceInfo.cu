#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("\nCUDA Device #%d\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f MB\n", prop.totalGlobalMem / 1048576.0);
        printf("Shared Memory Per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads Dim: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Size: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0);
        printf("Number of Multiprocessors: %d\n", prop.multiProcessorCount);
    }

    return 0;
}