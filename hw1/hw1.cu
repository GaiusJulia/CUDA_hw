#include<stdio.h>
#include<iostream>

// Задание 1: поэлементное перемножение векторов
__global__
void KernelMul(int n, float* x, float* y, float* res){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int index = tid; index < n; index += stride){
        res[index] = x[index] * y[index];
    }
}

int main() {
    //step 1
    int n = 1 << 20;
    float* h_x = new float[n];
    float* h_y = new float[n];
    float* h_res = new float[n];

    //step 2
    float* d_x;
    float* d_y;
    float* d_res;
    int nbytes = n * sizeof(float);
    cudaMalloc(&d_x, nbytes);
    cudaMalloc(&d_y, nbytes);
    cudaMalloc(&d_res, nbytes);

    //
    for (int i = 0; i < n; i++)
    {
        h_x[i] = 2 * i;
        h_y[i] = 3 * i;
    }

    //step 3
    cudaMemcpy(d_x, h_x, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //step 4
    cudaEventRecord(start);

    // первое число - число блоков, второе число - BLOCKSIZE (число потоков в блоке)
    KernelMul<<<1, 2>>>(n, d_x, d_y, d_res);

    cudaEventRecord(end);

    //step 5
    cudaMemcpy(h_res, d_res, nbytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    std::cout << "Time elapsed: " << ms << " ms " << std::endl;

    /*
    for (int i = 0; i < n; i++)
    {
        std::cout << h_res[i] << " ";
    }
    //*/

    //step 6
    delete[] h_x;
    delete[] h_y;
    delete[] h_res;
    //step 7
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
}
