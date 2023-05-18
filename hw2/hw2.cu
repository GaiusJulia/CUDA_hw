#include<stdio.h>
#include<iostream>

// Задание 2: Сложение двух матриц
__global__
void ArrSum(float* A, float* B, float* res, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int width = blockDim.y * gridDim.y;

    res[i * width + j] = A[i * width + j] + B[i * width + j];
}

int main() {
    //step 1
    int n = 128;
    int m = 64;
    float* h_A = new float[n*m];
    float* h_B = new float[n*m];
    float* h_res = new float[n*m];

    //step 2
    float* d_A;
    float* d_B;
    float* d_res;
    int nbytes = n * m * sizeof(float);
    cudaMalloc(&d_A, nbytes);
    cudaMalloc(&d_B, nbytes);
    cudaMalloc(&d_res, nbytes);

    //
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            h_A[i*m + j] = 1;
            h_B[i*m + j] = 2;
        }
    }

    //step 3
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //step 4
    cudaEventRecord(start);
    dim3 num_blocks(16, 16);
    dim3 block_size(8, 4);
    // необходимо num_blocks.x * block_size.x = n,  num_blocks.y * block_size.y = m
    ArrSum<<<num_blocks, block_size>>>(d_A, d_B, d_res, n * m);

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
        for (int j = 0; j < m; j++)
            std::cout << h_res[i*m + j] << " ";
        std::cout << std::endl;
    }
    //*/

    //step 6
    delete[] h_A;
    delete[] h_B;
    delete[] h_res;
    //step 7
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res);
}