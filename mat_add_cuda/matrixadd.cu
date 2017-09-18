/* The matrix addition example on CUDA
This program first reads two matrices from two text files namely matA.txt and matB.txt
Then it does matrix addition on the two matrices
Finally it saves the answer matrix in a text file named ans.txt
The program also measures the time taken for matrix operation on CUDA
*/

#include <stdio.h>

//define file names here
#define MATRIXA "matA.txt"
#define MATRIXB "matB.txt"
#define MATRIXANS "ans.txt"

//define the sizes of the matrices here
#define ROWS 16
#define COLS 16
#define SIZE ROWS*COLS

//kernel that does the matrix addition. Just add each element to the respective one
__global__ void addMatrix(int *ans_cuda,int *matA_cuda,int *matB_cuda){
	int row = threadIdx.y;
	int col = threadIdx.x;
	int position = row*COLS + col;
	ans_cuda[position]=matA_cuda[position]+matB_cuda[position];
}

int main(){
	
	//open the files 
	FILE *filematA = fopen(MATRIXA, "r");
	FILE *filematB = fopen(MATRIXB, "r");
	FILE *fileans = fopen(MATRIXANS, "w");
	
	//allocate matrices
	int matA[SIZE];
	int matB[SIZE];
	int ans[SIZE];
	
	//read the input matrices from file
	int row,col;
	for(row=0;row<ROWS;row++){
		for(col=0;col<COLS;col++){
			int position = row*COLS + col;
			fscanf(filematA, "%d", &matA[position]);
			fscanf(filematB, "%d", &matB[position]);
			ans[position]=0;
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

	//the moment at which we start measuring the time
	cudaEventCreate(&start);
	cudaEventRecord(start,0);
	
	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(int)*SIZE);
	cudaMalloc((void **)&matB_cuda,sizeof(int)*SIZE);	
	cudaMalloc((void **)&ans_cuda,sizeof(int)*SIZE);
		
	//copy contents from ram to cuda
	cudaMemcpy(matA_cuda, matA, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(matB_cuda, matB, sizeof(int)*SIZE, cudaMemcpyHostToDevice);	

	//thread configuration 
	dim3 numBlocks(1,1);
	dim3 threadsPerBlock(COLS,ROWS);
	
	//do the matrix addition on CUDA
	addMatrix<<<numBlocks,threadsPerBlock>>>(ans_cuda,matA_cuda,matB_cuda);

	//copy the answer back
	cudaMemcpy(ans, ans_cuda, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);	

	//the moment at which we stop measuring time 
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	
	//free the memory we allocated on CUDA
	cudaFree(matA_cuda);
	cudaFree(matB_cuda);
	cudaFree(ans_cuda);
	
/*************************CUDA STUFF ENDS HERE************************/


	//write the answer to the text file
	for(row=0;row<ROWS;row++){
		for(col=0;col<COLS;col++){
			int position = row*COLS + col;
			fprintf(fileans, "%d ", ans[position]);
		}
		fprintf(fileans, "\n");
	}
	
	//Find and print the elapsed time
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for operation is %.10f seconds\n",elapsedtime/(float)1000);

	fclose(filematA);
	fclose(filematB);
	fclose(fileans);
		
	return 0;
}