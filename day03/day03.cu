#define BLUR_SIZE 1
#include<stdio.h>

__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float val = 0.0;
    int num_pixels = 0;
    for(int i=-BLUR_SIZE; i<BLUR_SIZE+1; ++i){
        int cur_row = row + i;
        if(cur_row < 0 || cur_row >= height) continue;
        for(int j=-BLUR_SIZE; j<BLUR_SIZE+1; ++j){
            int cur_col = col + j;
            if(cur_col < 0 || cur_col >= width) continue;
            val += input[cur_row*width + cur_col];
            num_pixels++;
        }
    }
    output[row*width + col] = (unsigned char) val / num_pixels;
}


int main(){
    int width = 1280;
    int height = 1920;

    unsigned char *input, *output, *d_input, *d_output;

    input = (unsigned char*)malloc(width*height*sizeof(unsigned char));
    output = (unsigned char*)malloc(width*height*sizeof(unsigned char));

    for(int i=0; i<width*height; ++i){
        input[i] = i;
    }

    cudaMalloc((void**)&d_input, width*height*sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width*height*sizeof(unsigned char));

    cudaMemcpy(d_input, input, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

    int size = 16;
    dim3 threadsPerBlock(size, size);
    dim3 blocksPerGrid((width + size-1) / 16, (height + size-1) / 16);
    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    for(int i=0; i<10; ++i){
        printf("%d ", output[i]);
    }
    printf("\n");
    
    return 0;
}
