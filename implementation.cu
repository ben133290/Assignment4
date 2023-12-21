/*
============================================================================
Filename    : algorithm.c
Author      : Benedikt Heuser
SCIPER      : Your SCIPER number
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}

__constant__ int m0;
__constant__ int m1;
__constant__ int m2;
__constant__ int m3;

// GPU Optimized function
__global__ void kernel(double *input, double *output, int length) {
    // Calculate global indices
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int c = i * length + j;

    if (i < length - 1 && j < length - 1 && c != m0 && c != m1 && c != m2 && c != m3) {
        // Compute the convolution
        output[c] =
            (input[(i - 1) * length + (j - 1)] + input[(i - 1) * length + j] + input[(i - 1) * length + (j + 1)] +
             input[i * length + (j - 1)] + input[i * length + j] + input[i * length + (j + 1)] +
             input[(i + 1) * length + (j - 1)] + input[(i + 1) * length + j] + input[(i + 1) * length + (j + 1)]) /
            9.0;
    }


}

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    // set special values
    int m0v = (length/2-1)*length+(length/2-1);
    int m1v = (length/2)*length+(length/2-1);
    int m2v = (length/2-1)*length+(length/2);
    int m3v = (length/2)*length+(length/2);
    input[m0v] = 1000;
    input[m1v] = 1000;
    input[m2v] = 1000;
    input[m3v] = 1000;

    /* Preprocessing goes here */
    double *gpu_input, *gpu_output;
    
    cudaMalloc((void**)&gpu_input, length * length * sizeof(double));
    cudaMalloc((void**)&gpu_output, length * length * sizeof(double));

    cudaMemcpyToSymbol(m0, &m0v, sizeof(int));
    cudaMemcpyToSymbol(m1, &m1v, sizeof(int));
    cudaMemcpyToSymbol(m2, &m2v, sizeof(int));
    cudaMemcpyToSymbol(m3, &m3v, sizeof(int));

    dim3 threadsPerBlock(1, 1024);
    int blockSidex = length/threadsPerBlock.x;
    if (length % threadsPerBlock.x != 0) {blockSidex++;}
    int blockSidey = length/threadsPerBlock.y;
    if (length % threadsPerBlock.y != 0) {blockSidey++;}
    dim3 numOfBlocks(blockSidex, blockSidey);

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */
    cudaMemcpy(gpu_input, input, length * length * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    // Copy array from host to device <- I think this is misplaced but it was in the original ???
    cudaEventRecord(comp_start);

    /* GPU calculation goes here */
    for (int n = 0; n < iterations; n++) {

        kernel<<<numOfBlocks, threadsPerBlock>>>(gpu_input, gpu_output, length);

        if (n < iterations - 1) {
            double* temp = gpu_input;
            gpu_input = gpu_output;
            gpu_output = temp;
        }
    }

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);



    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */
    cudaMemcpy(output, gpu_output, length * length * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */
    cudaFree(gpu_input);
    cudaFree(gpu_output);

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}