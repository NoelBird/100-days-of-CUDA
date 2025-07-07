## Day06 - cudaGetDeviceCount

## Compilation

```
nvcc -o day06 day06.cu
```

## Result

```
root@8654901c2703:~/kernel# ./day06 
Device 0: Orin
Total Global Memory: 65893269504 bytes
Shared Memory per Block: 49152 bytes
Registers per Block: 65536
Warp Size: 32
Max Threads per Block: 1024
Max Threads Dim: (1024, 1024, 64)
Max Grid Size: (2147483647, 65535, 65535)
Clock Rate: 1300000 MHz
Total Constant Memory: 65536 bytes
Compute Capability: 8.7
Device Overlap: Yes
Multiprocessor Count: 8
Kernel Execution Timeout: No
Unified Addressing: Yes
Concurrent Kernels: Yes
ECC Enabled: No
Memory Clock Rate: 612000 MHz
Memory Bus Width: 256 bits
L2 Cache Size: 4194304 bytes
Texture Alignment: 512
Texture Pitch Alignment: 32
```