#include <stdio.h>

int main(){
    int devCount;
    cudaGetDeviceCount(&devCount);

    for(int i=0; i<devCount; ++i){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
        printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Warp Size: %d\n", prop.warpSize);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads Dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Clock Rate: %d MHz\n", prop.clockRate);
        printf("Total Constant Memory: %zu bytes\n", prop.totalConstMem);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Device Overlap: %s\n", prop.deviceOverlap ? "Yes" : "No");
        printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
        printf("Kernel Execution Timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("Memory Clock Rate: %d MHz\n", prop.memoryClockRate);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("L2 Cache Size: %zu bytes\n", prop.l2CacheSize);
        printf("Texture Alignment: %d\n", prop.textureAlignment);
        printf("Texture Pitch Alignment: %d\n", prop.texturePitchAlignment);
    }
    return 0;
}
