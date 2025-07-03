## Today Objective
- make a grayscale kernel

## Device
- Jetson Orin AGX (SM87)

## Compile Option
- nvcc -arch=sm_87 day02.cu -o day02

## Result

```
root@8654901c2703:~/kernel# ./day02 
103 198 105 
115 81 255 
74 236 41 
205 186 171 
242 251 227 
70 124 194 
84 248 27 
232 231 141 
118 90 46 
99 51 159 


171 100 187 188 247 117 197 224 92 69
```
