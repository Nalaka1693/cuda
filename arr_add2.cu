#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>

#define SIZE 10

__global__ void fillArr(int arr[]);

__global__ void opArr(int arr1[], double arr2[]);

int main() {
    int i = 0;
    time_t t;
    srand((unsigned) time(&t));

    int arr1[SIZE];
    double arr2[(int) SIZE / 2];

    for (i = 0; i < SIZE; ++i) {
        arr1[i] = rand() % SIZE;
    }

    int *dArr1;
    double *dArr2;
    cudaMalloc((void **) &dArr1, sizeof(int) * SIZE);
    cudaMalloc((void **) &dArr2, sizeof(double) * SIZE);

    cudaMemcpy(dArr1, arr1, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
    opArr << < i, SIZE / 2 >> > (dArr1, dArr2);
    cudaMemcpy(arr2, dArr2, sizeof(double) * SIZE, cudaMemcpyDeviceToHost);

    for (i = 0; i < (int) SIZE / 2; ++i) {
        printf("%4d     %.2f\n", i, arr2[i]);
    }
    printf("\n");

    return 0;
}

__global__ void opArr(int arr1[], double arr2[]) {
    int i = threadIdx.x;

    arr2[i] = (arr1[2 * i] + arr2[2 * i + 1]) / 2;
}
