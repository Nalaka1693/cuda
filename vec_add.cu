/* Program to add two vectors in CUDA
Last vector addition program only works for vectors less than 1024 elements
The reason is last time we only configured the kernel to run one block
As maximum number of threads per block is CUDA is 1024 it gave wrong answers for large vectors
So this time we make use of multiple blocks to make the program work for vectors larger than 1024 elements
*/

#include <stdio.h>
#include <math.h>

#define SIZE 2048 * 64

__global__ void sumVector1(int *vectorAns_cuda, int *vectorA_cuda);

int main() {

    //arrays in main memory
    int vectorA[SIZE];
    int vectorB[SIZE];
    int vectorAns[SIZE];

    //generate some values
    int i;
    for (i = 0; i < SIZE; i++) {
        vectorA[i] = i;
        vectorB[i] = SIZE - i;
    }

    //pointers for arrays to be put on cuda memory
    int *vectorA_cuda;
    int *vectorB_cuda;
    int *vectorAns_cuda;

    //allocate memory in cuda device
    cudaMalloc((void **) &vectorA_cuda, sizeof(int) * SIZE);
    cudaMalloc((void **) &vectorAns_cuda, sizeof(int) * SIZE / 2);

    //copy contents from main memory to cuda device memory
    cudaMemcpy(vectorA_cuda, vectorA, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    //thread configuration
    /* Here we arbitrarily specify 256 threads per block
       Then we divide the number of threads by 256 and round it off to the next int to get the number of blocks needed
    */
    int numBlocks = ceil(SIZE / (float) 256);
    int threadsPerBlock = 256;

    //call the cuda kernel
    sumVector1 <<< numBlocks, threadsPerBlock >>> (vectorAns_cuda, vectorA_cuda, vectorB_cuda);

    //copy back the results from cuda memory to main memory
    cudaMemcpy(vectorAns, vectorAns_cuda, sizeof(int) * SIZE / 2, cudaMemcpyDeviceToHost);

    //print the answer
    printf("Answer is : ");

    for (i = 0; i < SIZE; i++) {
        printf("%d ", vectorAns[i]);
    }
    printf("\n");

    return 0;
}

/** CUDA kernel to add two vectors*/
__global__ void sumVector1(int *vectorAns_cuda, int *vectorA_cuda) {

    //threadIndex = blockSize * blockIndex + threadIndex
    /* Here we have to calculate the position of the element in the array by using threadIndex, blockIndex and block Size*/
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    //if the SIZE id not divisible by 256 there would be threads that are out of range the the array
    // such threads should idle rather than doing any work
    // So using a if condition we prevent them from doing any operation
    if (tid < SIZE / 2) {
        vectorAns_cuda[tid] = (vectorA_cuda[2 * tid] + vectorB_cuda[2 * tid + 1]) / 2;
    }
}

