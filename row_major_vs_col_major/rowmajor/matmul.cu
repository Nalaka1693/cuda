/* Program to do matrix multiplication in cuda
This program generates two matrices and multiply them
Uses row major flattening of 2D arrays
Prints the time taken for operations
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "helpers.cuh"

//Dimensions for matrix1
#define ROWS1 1000
#define COLS1 2000

//DImensions for matrix2
#define ROWS2 2000
#define COLS2 1500

/** CUDA kernel to do matrix mutiplication**/
__global__ void matMul(int *matC_cuda, int *matA_cuda, int *matB_cuda){
	
	//derive the row and column based on thread configuration
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	
	//Limit calculations for valid indices
	if(row < ROWS1 && col < COLS2){
	
		int prod=0;
		int k;
		for(k=0;k<COLS1;k++){
			prod=prod+matA_cuda[row*COLS1+k]*matB_cuda[k*COLS2+col];
		}
		matC_cuda[row*COLS2+col]=prod;	
		
	}
	
}

int main(){
	
	//check whether dimensions are valid for matrix mutiplication
	if(COLS1!=ROWS2){
		printf("Matrix dimensions are invalid for matrix multiplication\n");
		exit(1);
	}
	
	//Initialize arrays in RAM
	int *matA = (int *)malloc(sizeof(int)*ROWS1*COLS1);
	int *matB = (int *)malloc(sizeof(int)*ROWS2*COLS2);
	int *matC = (int *)malloc(sizeof(int)*ROWS1*COLS2);	
	//check if out of memory.
	if(matA==NULL || matB==NULL || matC==NULL){
		perror("Memory out");
		exit(EXIT_FAILURE);
	}	
	
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

	/********************************** CUDA stuff starts here *******************************/
	
	//start meauring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
	//pointers for memory allocation in cudaa
	int *matA_cuda;
	int *matB_cuda;
	int *matC_cuda;
	
	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(int)*ROWS1*COLS1);
	checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(int)*ROWS2*COLS2);
	checkCudaError();
	cudaMalloc((void **)&matC_cuda,sizeof(int)*ROWS1*COLS2);
	checkCudaError();
	
	//copy memory from ram to cuda
	cudaMemcpy(matA_cuda,matA,sizeof(int)*ROWS1*COLS1,cudaMemcpyHostToDevice);
	checkCudaError();
	cudaMemcpy(matB_cuda,matB,sizeof(int)*ROWS2*COLS2,cudaMemcpyHostToDevice);
	checkCudaError();
	
	//multiply the matrices 
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(ceil(COLS2/(float)16),ceil(ROWS1/(float)16));
	
	//start measuring time for cuda kernel only
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);	
	
	matMul<<<numBlocks,threadsPerBlock>>>(matC_cuda,matA_cuda,matB_cuda);
	cudaDeviceSynchronize(); checkCudaError();

	//end measuring time for cuda kernel
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);
		
	//copy the answer back from cuda ro ram
	cudaMemcpy(matC,matC_cuda,sizeof(int)*ROWS1*COLS2,cudaMemcpyDeviceToHost);
	checkCudaError();

	//free the cuda memory
	cudaFree(matA_cuda);
	checkCudaError();
	cudaFree(matB_cuda);
	checkCudaError();
	cudaFree(matC_cuda);
	checkCudaError();
	
	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	
	/********************** CUDA stuff ends here ********************************/
	
	//print the answer
	printf("Answer : \n");	
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%5d ",matC[i*COLS2+j]);
		}
		printf("\n");
	}	
	
	//print the time spent to stderr
	fprintf(stderr,"Time spent for CUDA kernel is %1.5f seconds\n",elapsedtimekernel/(float)1000); 
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 

	return 0;
}
