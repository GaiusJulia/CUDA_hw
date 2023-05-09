#include<stdio.h>
#include<iostream>

// Задание 3: Перемножение матрицы на вектор
__global__
void ArrVectMul(float* A, float* x, float* res, int size){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    res[i] = 0.;
    for (int k = 0; k < size; k++)
    {
        res[i] += A[i * size + k] * x[k];
        //printf("A[%d][%d] = %f \n", i, k, A[i * size + k]);
        //printf("x[%d] = %f \n", k, x[k]);
    }

}

int main() {
    int n = 10;
    int m = 20;
    float* h_A = new float[n*m];
    float* h_x = new float[m];
    float* h_res = new float[n];

    //step 2
    float* d_A;
    float* d_x;
    float* d_res;
    int bytes = sizeof(float);
    cudaMalloc(&d_A, n * m * bytes);
    cudaMalloc(&d_x, m * bytes);
    cudaMalloc(&d_res, n * bytes);

    //
    for (int i = 0; i < m; i++)
    {
        h_x[i] = 1;

        for (int j = 0; j < n; j++)
        {
            h_A[j*m + i] = j;
        }
    }

    //step 3
    cudaMemcpy(d_A, h_A, n * m * bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, m *bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    //step 4
    cudaEventRecord(start);
    int num_blocks = 16;
    int block_size = 16;
    ArrVectMul<<<num_blocks, block_size>>>(d_A, d_x, d_res, m);

    cudaEventRecord(end);

    //step 5
    cudaMemcpy(h_res, d_res, n * bytes, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(end);

    float ms;
    cudaEventElapsedTime(&ms, start, end);
    std::cout << "Time elapsed: " << ms << " ms " << std::endl;

    
    for (int i = 0; i < n; i++)
    {
        std::cout << h_res[i] << " ";
    }

    //step 6
    delete[] h_A;
    delete[] h_x;
    delete[] h_res;
    //step 7
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_res);
}
