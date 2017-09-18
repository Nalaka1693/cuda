/* Program to add two vectors in CUDA 
Last vector addition program only works for vectors less than 1024 elements
The reason is last time we only configured the kernel to run one block
As maximum number of threads per block is CUDA is 1024 it gave wrong answers for large vectors
So this time we make use of multiple blocks to make the program work for vectors larger than 1024 elements
*/

#include <stdio.h>
#include <math.h>
#include "helpers.cuh"

#define SIZE 2000

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
	checkCuda(cudaMalloc((void **)&vectorA_cuda,sizeof(int)*SIZE));
	checkCuda(cudaMalloc((void **)&vectorB_cuda,sizeof(int)*SIZE));		
	checkCuda(cudaMalloc((void **)&vectorAns_cuda,sizeof(int)*SIZE));
	
	//copy contents from main memory to cuda device memory
	checkCuda(cudaMemcpy(vectorA_cuda,vectorA,sizeof(int)*SIZE,cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(vectorB_cuda,vectorB,sizeof(int)*SIZE,cudaMemcpyHostToDevice));
	
	//thread configuration
	/* Here we arbitrarily specify 256 threads per block
		Then we divide the number of threads by 256 and round it off to the next int to get the number of blocks needed
		*/
		
	int threadsPerBlock = 256;	
	int numBlocks = ceil(SIZE/(float)threadsPerBlock);

	//call the cuda kernel
	addVector<<<numBlocks,threadsPerBlock>>>(vectorAns_cuda, vectorA_cuda, vectorB_cuda);
	checkCuda(cudaGetLastError());
	
	//copy back the results from cuda memory to main memory
	checkCuda(cudaMemcpy(vectorAns,vectorAns_cuda,sizeof(int)*SIZE,cudaMemcpyDeviceToHost));
	
	//print the answer
	printf("Answer is : ");

	for(i=0;i<SIZE;i++){
		printf("%d ",vectorAns[i]);
	}
	
	return 0;
}

/** CUDA kernel to add two vectors*/
__global__  void addVector(int *vectorAns_cuda, int *vectorA_cuda, int *vectorB_cuda ){
	
	//threadIndex = blockSize * blockIndex + threadIndex
	/* Here we have to calculate the position of the element in the array by using threadIndex, blockIndex and block Size
	*/
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	
	//if the SIZE id not divisible by 256 there would be threads that are out of range the the array
	// such threads should idle rather than doing any work
	// So using a if condition we prevent them from doing any operation
	if(tid<SIZE){
		int i,j,k;
		for(i=0;i<10;i++){
			for(j=0;j<10;j++){
					for(k=0;k<10;k++){
						if(tid<k){
							vectorAns_cuda[tid]=(int)(sinf((double)vectorAns_cuda[tid])+vectorA_cuda[tid]+vectorB_cuda[tid]);
						}
						else if (tid<j){
							vectorAns_cuda[tid]=(int)(cosf(vectorAns_cuda[tid])+vectorA_cuda[tid]-vectorB_cuda[tid]);
						}
						else{
							vectorAns_cuda[tid]=(int)(tanf(vectorAns_cuda[tid]-vectorA_cuda[tid]*vectorB_cuda[tid]));
						}
					}
			}
		}	
				
	}

}