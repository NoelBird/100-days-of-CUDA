## Device
- Jetson Orin AGX (SM87)

## Compile Option
- nvcc -arch=sm_87 day1.cu -o day1

## Mistakes I made
- kernel launching size should not be <<<1, 1>>>. because I want to add vector values
  - all results are 0 
  - to check if compile error, I added `cudaGetLastError` 

## C-style or C++ style?
- device variables such as d_a should be dA?
  - I selected PascalCase

## Result

```text
First 10 results:
0 + 0 = 0
1 + 1 = 2
2 + 2 = 4
3 + 3 = 6
4 + 4 = 8
5 + 5 = 10
6 + 6 = 12
7 + 7 = 14
8 + 8 = 16
9 + 9 = 18

Last 10 results:
1014 + 1014 = 2028
1015 + 1015 = 2030
1016 + 1016 = 2032
1017 + 1017 = 2034
1018 + 1018 = 2036
1019 + 1019 = 2038
1020 + 1020 = 2040
1021 + 1021 = 2042
1022 + 1022 = 2044
1023 + 1023 = 2046
```
