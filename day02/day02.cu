#include <stdio.h>
#define CHANNELS 3

__global__ void colorToGrayscaleConversion(unsigned char* input, unsigned char* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int grayOffset = row * width + col;
    int rgbOffset = grayOffset*CHANNELS;
    unsigned char r = input[rgbOffset];
    unsigned char g = input[rgbOffset+1];
    unsigned char b = input[rgbOffset+2];

    unsigned char gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    
    output[grayOffset] = gray;
}

int main() {
    int width = 1920;
    int height = 1080;

    unsigned char* input = new unsigned char[width * height * CHANNELS];
    unsigned char* output = new unsigned char[width * height];
    unsigned char* d_input;
    unsigned char* d_output;

    for (int i = 0; i < width * height * CHANNELS; i++) {
        input[i] = rand() % 256;
    }

    for (int i = 0; i < width * height; i++) {
        output[i] = 0;
    }

    for (int i = 0; i < 10; i++) {
        printf("%d ", input[i*CHANNELS]);
        printf("%d ", input[i*CHANNELS+1]);
        printf("%d ", input[i*CHANNELS+2]);
        printf("\n");
    }
    printf("\n\n");
    
    cudaMalloc((void**)&d_input, width * height * CHANNELS * sizeof(unsigned char));
    cudaMalloc((void**)&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, input, width * height * CHANNELS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    colorToGrayscaleConversion<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    for (int i = 0; i < 10; i++) {
        printf("%d ", output[i]);
    }
    printf("\n\n");

    delete[] input;
    delete[] output;

    return 0;
}
