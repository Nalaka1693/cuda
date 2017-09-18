#include <stdio.h>
#include "helpers.cuh"

#define SIZE 2048

__global__  void addVector(int *vectorAns_cuda, int *vectorA_cuda, int *vectorB_cuda );

int main(){

	//arrays in main memory
	int vectorA[SIZE];
	int vectorB[SIZE];
	int vectorAns[SIZE];
	
	//generate some values
	int i;
	for(i=0;i<SIZE;i++){
		vectorA[i]=i;
		vectorB[i]=SIZE-i;
	}
	
	//pointers for arrays to be put on cuda memory
	int *vectorA_cuda;
	int *vectorB_cuda;
	int *vectorAns_cuda;
	
	//allocate memory in cuda device
	cudaMalloc((void **)&vectorA_cuda,sizeof(int)*SIZE);
	checkCudaError();
	cudaMalloc((void **)&vectorB_cuda,sizeof(int)*SIZE);
	checkCudaError();	
	cudaMalloc((void **)&vectorAns_cuda,sizeof(int)*SIZE);
	checkCudaError();
	
	//copy contents from main memory to cuda device memory
	cudaMemcpy(vectorA_cuda,vectorA,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
	checkCudaError();
	cudaMemcpy(vectorB_cuda,vectorB,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
	checkCudaError();
	
	//call the cuda kernel
	addVector<<<1,SIZE>>>(vectorAns_cuda, vectorA_cuda, vectorB_cuda);
	checkCudaError();
	//now it will complain here and will exit the program
	
	//copy back the results from cuda memory to main memory
	cudaMemcpy(vectorAns,vectorAns_cuda,sizeof(int)*SIZE,cudaMemcpyDeviceToHost);
	checkCudaError();
	
	printf("Answer is : ");

	for(i=0;i<SIZE;i++){
		printf("%d ",vectorAns[i]);
	}
	
	return 0;
}

__global__  void addVector(int *vectorAns_cuda, int *vectorA_cuda, int *vectorB_cuda ){
	
	int tid=threadIdx.x;
	vectorAns_cuda[tid]=vectorA_cuda[tid]+vectorB_cuda[tid];

}