## Today Objective
- make a matmul kernel

## Device
- Jetson Orin AGX (SM87)

## Compile Option
- nvcc -arch=sm_87 day04.cu -o day04

## Result

```
root@b2d9006ed163:~/kernels# ./simple_matmul 
First 10 results:
C[0] = 56.000000
C[1] = 62.000000
C[2] = 68.000000
C[3] = 74.000000
C[4] = 152.000000
C[5] = 174.000000
C[6] = 196.000000
C[7] = 218.000000
C[8] = 248.000000
C[9] = 286.000000

Verification (first element):
Expected C[0] = 56.000000, Got C[0] = 56.000000
```

