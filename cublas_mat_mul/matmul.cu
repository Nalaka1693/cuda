/* Program to do matrix multiplication using CUBLAS library
 * Note that CUBLAS needs matrices to be placed in column major order
 * More help at http://docs.nvidia.com/cuda/cublas/#axzz3VhSH0Yvj
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "helpers.cuh"
#include <cublas_v2.h>

//Dimensions for matrix1
#define ROWS1 3000
#define COLS1 2000

//DImensions for matrix2
#define ROWS2 2000
#define COLS2 3000


int main(){
	
	//check whether dimensions are valid for matrix mutiplication
	if(COLS1!=ROWS2){
		printf("Matrix dimensions are invalid for matrix multiplication\n");
		exit(1);
	}
	
	//Initialize arrays in RAM
	float *matA = (float *)malloc(sizeof(float)*ROWS1*COLS1);
	float *matB = (float *)malloc(sizeof(float)*ROWS2*COLS2);
	float *matC = (float *)malloc(sizeof(float)*ROWS1*COLS2);	
	//check if out of memory.
	if(matA==NULL || matB==NULL || matC==NULL){
		perror("Memory out");
		exit(EXIT_FAILURE);
	}	
	
	//generate some values for matrixA : CUBLAS needs matrices to be placed in column major order
	int i,j;
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			matA[j*ROWS1+i]=i+j;
		}
	}

	//print the matA
	printf("Matrix A : \n");
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			printf("%f ",matA[j*ROWS1+i]);
		}
		printf("\n");
	}		
	printf("\n");

	
	//generate values for matrixB : CUBLAS needs matrices to be placed in column major order
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			matB[j*ROWS2+i]=i-j;
		}
	}

	//print the matB
	printf("Matrix B : \n");
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			printf("%f ",matB[j*ROWS2+i]);
		}
		printf("\n");
	}	
	printf("\n");

	/********************************** CUDA stuff starts here *******************************/
	
	//start measuring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
	//pointers for memory allocation in cudaa
	float *matA_cuda;
	float *matB_cuda;
	float *matC_cuda;
	
	//allocate memory in cuda
	cudaMalloc((void **)&matA_cuda,sizeof(float)*ROWS1*COLS1); checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(float)*ROWS2*COLS2); checkCudaError();
	cudaMalloc((void **)&matC_cuda,sizeof(float)*ROWS1*COLS2); checkCudaError();
	
	//copy memory from ram to cuda
	cudaMemcpy(matA_cuda,matA,sizeof(float)*ROWS1*COLS1,cudaMemcpyHostToDevice); checkCudaError();
	cudaMemcpy(matB_cuda,matB,sizeof(float)*ROWS2*COLS2,cudaMemcpyHostToDevice); checkCudaError();
	
	//start measuring time for cuda kernel only
	cudaEvent_t startkernel,stopkernel;
	float elapsedtimekernel;
	cudaEventCreate(&startkernel);
	cudaEventRecord(startkernel,0);	

	/********************************** CUBLAS *********************************/
	
	//cublas create a handle
	cublasHandle_t handle;
	checkCublasError(cublasCreate(&handle));

	/* Now the cublas multiplication must be called with proper arguments such that
	 * cublasSgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc)
	 * the description of arguments is as follows
	 * handle - handle to the cuBLAS library context.
	 * transa - whether the matA must be transposed before being multiplied : set CUBLAS_OP_N if no transposition , set CUBLAS_OP_T if need to be transposed
	 * transb - whether the matA must be transposed before being multiplied
	 * m - No of rows in the matA
	 * n - No of columns in matB
	 * k - No of columns in matA or No of rows in matB
	 * alpha - scalar which should be 1.0
	 * A - the pointer for the matA in device
	 * lda - leading dimension of two-dimensional array used to store the matA. This is usually the number of rows in matA
	 * B - the pointer for the matB in device
	 * ldb - leading dimension of two-dimensional array used to store the matB. This is usually the number of rows in matB
	 * C - the pointer for the matC in device
	 * ldc - leading dimension of two-dimensional array used to store the matC. This is usually the number of rows in matC
	 */
	
	//calculation is such that :  matC = alpha*matA*matB + beta*matC 	
	//Therefore need to set alpha=1 and beta=0 to get matC = matA*matB 
	const float alpha = 1.0;
	const float beta = 0.0;
	 
	//call the function
	checkCublasError(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,ROWS1,COLS2,COLS1,&alpha,matA_cuda,ROWS1,matB_cuda,ROWS2,&beta,matC_cuda,ROWS1));

	//destroy the handler
	checkCublasError(cublasDestroy(handle));
	
	/******************************* CUBLAS END *********************************/
	
	//end measuring time for cuda kernel
	cudaEventCreate(&stopkernel);
	cudaEventRecord(stopkernel,0);
	cudaEventSynchronize(stopkernel);
	cudaEventElapsedTime(&elapsedtimekernel,startkernel,stopkernel);
		
	//copy the answer back from cuda ro ram
	cudaMemcpy(matC,matC_cuda,sizeof(float)*ROWS1*COLS2,cudaMemcpyDeviceToHost); checkCudaError();

	//free the cuda memory
	cudaFree(matA_cuda); checkCudaError();
	cudaFree(matB_cuda); checkCudaError();
	cudaFree(matC_cuda); checkCudaError();
	
	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	
	/********************** CUDA stuff ends here ********************************/
	
	//print the answer : CUBLAS gives the answer placed in column major order
	printf("Answer : \n");	
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%f ",matC[j*ROWS1+i]);
		}
		printf("\n");
	}	
	
	//print the time spent to stderr
	fprintf(stderr,"Time spent for CUBLAS operation is %1.5f seconds\n",elapsedtimekernel/(float)1000); 
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 

	return 0;
}
