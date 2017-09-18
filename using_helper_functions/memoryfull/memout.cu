/* This example shows how wrong results would come when memory sizes which are
beyond the capacity of the graphics card is allocated
By default it will not complain and to prevent such errors
you should check the return value of cudaMalloc() functions by using the
helper functions provided*/

#include <stdio.h>
#include "helpers.cuh"

/* actual size of the vector is 1024. But to demonstrate the things we allocate a larger array than this */
#define SIZE 204800000000
#define REALSIZE 1024

__global__  void cudawork(int *vectorA_cuda);

int main(){

	//pointers for arrays to be put on cuda memory
	int *A_cuda;

	//allocate memory in cuda device
	cudaMalloc((void **)&A_cuda,sizeof(int)*SIZE);
	checkCudaError();
	//now it will complain here and will exit
	
	//call cuda kernel
	cudawork<<<1,REALSIZE>>>(A_cuda);
	checkCudaError();;
	
	//allocate memory in RAM and copy them
	int *A= (int *)malloc(sizeof(int)*REALSIZE);
	
	//copy the things from cuda to ram
	cudaMemcpy(A, A_cuda, sizeof(int)*REALSIZE,cudaMemcpyDeviceToHost);
	checkCudaError();
	
	//print the result.
	//If correctly executed all 1024 elements should be set to 1
	int i=0;
	for(i=0;i<REALSIZE;i++){
		printf("%d ",A[i]);
	}
	
	return 0;
}

/* A kernel that set the value of each slot to 1*/
__global__  void cudawork(int *vectorA_cuda){
	
	int tid=threadIdx.x;
	vectorA_cuda[tid]=1;

}