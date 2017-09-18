/* Matrix multiplication program for CPU
This program generates a matrix of defined size and multiplies them
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "helpers.cuh"

//Dimensions of the first matrix
#define ROWS1 16
#define COLS1 16

//DImensions of the seconds matrix
#define ROWS2 16
#define COLS2 16

/* Function to do matriix multiplication */
__global__ void matMul(int* matC, int* matA, int* matB){
        int row,col,k, prod;
        row = blockIdx.y * blockDim.y + threadIdx.y;
        col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < ROWS1 && col < COLS2) {
           for(k=0;k<COLS1;k++){
                prod=prod+matA[row*COLS1+k]*matB[k*COLS2+col];
           }
           matC[row*COLS2+col]=prod;
        }

}

int main(){

        //check whether dimensions are valid for a multiplication
        if(COLS1!=ROWS2){
           printf("Matrix dimensions are invalid for matrix multiplication\n");
           exit(1);
        }

        //Initialize arrays in RAM
        int matA[ROWS1 * COLS1];
        int matB[ROWS2 * COLS2];
        int matC[ROWS1 * COLS2];

        //generate some values for matrixA
        int i,j;
        for(i=0;i<ROWS1;i++){
           for(j=0;j<COLS1;j++){
                matA[i*COLS1+j]=i+j;
           }
        }

        //print the matA
        printf("Matrix A : \n");
        for(i=0;i<ROWS1;i++){
           for(j=0;j<COLS1;j++){
                printf("%5d ",matA[i*COLS1+j]);
           }
           printf("\n");
        }
        printf("\n");

        //generate values for matrixB
        for(i=0;i<ROWS2;i++){
           for(j=0;j<COLS2;j++){
                matB[i*COLS2+j]=i-j;
           }
        }

        //print the matB
        printf("Matrix B : \n");
        for(i=0;i<ROWS2;i++){
           for(j=0;j<COLS2;j++){
                printf("%5d ",matB[i*COLS2+j]);
           }
           printf("\n");
        }
        printf("\n");

        /************cuda**********/
        int* cudaMatA;
        int* cudaMatB;
        int* cudaMatC;

        cudaMalloc((void **)&cudaMatA, sizeof(int)*ROWS1 * COLS1); checkCudaError();
        cudaMalloc((void **)&cudaMatB, sizeof(int)*ROWS2 * COLS2); checkCudaError();
        cudaMalloc((void **)&cudaMatC, sizeof(int)*ROWS1 * COLS2); checkCudaError();

        cudaMemcpy(cudaMatA, matA, sizeof(int)*ROWS1 * COLS1, cudaMemcpyHostToDevice); checkCudaError();
        cudaMemcpy(cudaMatB, matB, sizeof(int)*ROWS2 * COLS2, cudaMemcpyHostToDevice); checkCudaError();

        dim3 blockNum(ceil(COLS2/(float)16), ceil(ROWS1/(float)16));
        dim3 threadsPerBlocks(16, 16);


        clock_t start = clock();
        matMul<<<blockNum, threadsPerBlocks>>>(cudaMatC, cudaMatA, cudaMatB); checkCudaError();
        clock_t stop = clock();

        cudaMemcpy(matC, cudaMatC, sizeof(int)*ROWS1 * COLS2, cudaMemcpyDeviceToHost); checkCudaError();

        cudaFree(cudaMatA); checkCudaError();
        cudaFree(cudaMatB); checkCudaError();
        cudaFree(cudaMatC);     checkCudaError();

        //print the answer
        printf("Answer : \n");
        for(i=0;i<ROWS1;i++){
           for(j=0;j<COLS2;j++){
                printf("%5d ",matC[i*COLS2+j]);
           }
           printf("\n");
        }

        //calculate the time taken and print to stderr
        double elapsedtime = (stop-start)/(double)CLOCKS_PER_SEC;
        fprintf(stderr,"Elapsed time for operation on CPU is %1.5f seconds \n",elapsedtime);

        return 0;

}
