#include <stdio.h>

#define SIZE 8

__global__ void addVector(int vectorAns[SIZE], int vectorA[SIZE], int vectorB[SIZE]);

int main() {

    int vectorA[SIZE];
    int vectorB[SIZE];
    int vectorAns[SIZE];

    int i;
    for (i = 0; i < SIZE; i++) {
        vectorA[i] = i;
        vectorB[i] = SIZE - i;
    }

    int *d_A;
    int *d_B;
    int *d_C;

    //allocate memmory
    cudaMalloc((void **) &d_A, sizeof(int) * SIZE);
    cudaMalloc((void **) &d_B, sizeof(int) * SIZE);
    cudaMalloc((void **) &d_C, sizeof(int) * SIZE);

    //copy inputs from RAM to GPU
    cudaMemcpy(d_A, vectorA, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, vectorB, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    //calculation function
    addVector<<< 1, SIZE >>> (d_C, d_A, d_B);

    //copy back to RAM
    cudaMemcpy(vectorAns, d_C, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    //cuda free
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    printf("Answer is : \n");

    for (i = 0; i < SIZE; i++) {
        printf("%d ", vectorAns[i]);
    }
    printf("\n");

    return 0;
}

__global__ void addVector(int vectorAns[SIZE], int vectorA[SIZE], int vectorB[SIZE]) {
    int i = threadIdx.x;
    vectorAns[i] = vectorA[i] + vectorB[i];

}
