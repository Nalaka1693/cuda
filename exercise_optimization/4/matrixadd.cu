/* The matrix addition example on CUDA
*/

#include <stdio.h>
#include "helpers.cuh"

//define the sizes of the matrices here
#define ROWS 10000
#define COLS 10000
#define SIZE ROWS*COLS

//kernel that does the matrix addition. Just add each element to the respective one
__global__ void addMatrix(int *ans_cuda,int *matA_cuda,int *matB_cuda){
	
	/*blockDim.y gives the height of a block along y axis
	  blockDim.x gives the width of a block along x axis
	  blockIdx.y gives the index of the current block along the y axis
	  blockIdx.x gives the index of the current block along the x axis
	  threadIdx.y gives the index of the current thread in the current block along y axis
	  threadIdx.x gives the index of the current thread in the current block along x axis
	  */
	
	//calculate the row number based on block IDs and thread IDs
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	//calculate the column number based on block IDs and thread IDs
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	//to remove any indices beyond the size of the array
	if (row<ROWS && col <COLS){
		
		//conversion of 2 dimensional indices to single dimension
		int position = row*COLS + col;
	
		//do the calculation
		int a=matA_cuda[position];
		int b=matB_cuda[position];
		ans_cuda[position]=(int)a*b-a-b-a/(double)b-b/(double)a;
	
	}
}
int main(){
	
	//allocate matrices
	int *matA = (int *)malloc(sizeof(int)*SIZE);
	int *matB = (int *)malloc(sizeof(int)*SIZE);
	int *ans =(int *)malloc(sizeof(int)*SIZE);
	if(matA==NULL || matB==NULL || ans==NULL){
		perror("Mem full");
		exit(1);
	}
	
	//generate
	int row,col;
	for(row=0;row<ROWS;row++){
		for(col=0;col<COLS;col++){
			int position = row*COLS + col;
			matA[position]=row+col;
			matB[position]=row*col;
		}
	}
	

/*************************CUDA STUFF STARTS HERE************************/	
	
	//variables for time measurements
	cudaEvent_t start,stop;
	float elapsedtime;
	
	//pointers for cuda memory locations
	int *matA_cuda;
	int *matB_cuda;
	int *ans_cuda;	
	
	//allocate memory in cuda
	checkCuda(cudaMalloc((void **)&matA_cuda,sizeof(int)*SIZE));
	checkCuda(cudaMalloc((void **)&matB_cuda,sizeof(int)*SIZE));	
	checkCuda(cudaMalloc((void **)&ans_cuda,sizeof(int)*SIZE));
		
	//copy contents from ram to cuda
	checkCuda(cudaMemcpy(matA_cuda, matA, sizeof(int)*SIZE, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(matB_cuda, matB, sizeof(int)*SIZE, cudaMemcpyHostToDevice));	

	//thread configuration 
	int blockwidth=1;
	dim3 numBlocks(ceil(COLS/(float)blockwidth),ceil(ROWS/(float)blockwidth));
	dim3 threadsPerBlock(blockwidth,blockwidth);
	
	//do the matrix addition on CUDA
	//the moment at which we start measuring the time
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	addMatrix<<<numBlocks,threadsPerBlock>>>(ans_cuda,matA_cuda,matB_cuda);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	
	
	//the moment at which we stop measuring time 
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	//copy the answer back
	checkCuda(cudaMemcpy(ans, ans_cuda, sizeof(int)*SIZE, cudaMemcpyDeviceToHost));	

	//free the memory we allocated on CUDA
	checkCuda(cudaFree(matA_cuda));
	checkCuda(cudaFree(matB_cuda));
	checkCuda(cudaFree(ans_cuda));
	
/*************************CUDA STUFF ENDS HERE************************/


	/*write the answer
	for(row=0;row<ROWS;row++){
		for(col=0;col<COLS;col++){
			int position = row*COLS + col;
			printf("%5d ", ans[position]);
		}
		printf("\n");
	}*/
	
	//Find and print the elapsed time
	cudaEventElapsedTime(&elapsedtime,start,stop);
	fprintf(stderr,"Time spent for operation is %.10f seconds\n",elapsedtime/(float)1000);
		
	return 0;
}