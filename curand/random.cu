/*Generate random floating point matrix using CUDA using curand library */

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include "helpers.cuh"

#define ROWS 100
#define COLS 100
#define SEED 0
#define FILENAME "output.txt"


int main(){

	float *matrix, *matrix_cuda;

	//memory allocation in ram
	matrix=(float *)malloc(sizeof(float)*COLS*ROWS);
	if(matrix==NULL){
		fprintf(stderr,"RAM full\n");
		exit(1);
	}
	
	//memory allocation in cuda
	cudaMalloc((void **)&matrix_cuda,sizeof(float)*COLS*ROWS); checkCudaError();
	
	/*create a random number generator.
	Here CURAND_RNG_PSEUDO_DEFAULT is the type of the generator
	There are various other types*/
	curandGenerator_t generator;	
	checkCurandError(curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_DEFAULT));
	//set the seed
	checkCurandError(curandSetPseudoRandomGeneratorSeed(generator,SEED));
	
	//Time measurement
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
	/*generate random numbers in a uniform distribution
	There are various other distributions given by functions
	curandGenerateNormal, curandGenerateLogNormal, curandGeneratePoisson
	If you are generating double values use
	curandGenerateUniformDouble, curandGenerateNormalDouble etc*/
	checkCurandError(curandGenerateUniform(generator,matrix_cuda,COLS*ROWS));
	
	//time end
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	printf("Time spent for random number generation is : %.10f s\n",elapsedtime/(float)1000);	
	
	//memory copy 
	cudaMemcpy(matrix,matrix_cuda,sizeof(float)*COLS*ROWS,cudaMemcpyDeviceToHost); checkCudaError();
	
	//writing to file
	FILE *fp=fopen(FILENAME,"w");
	if(fp==NULL){
		fprintf(stderr,"Cannot open file for writing\n");
		exit(1);
	}
	
	int i,j;
	for (i=0;i<ROWS;i++){
		for (j=0;j<COLS;j++){
			fprintf(fp,"%f ",matrix[i*COLS+j]*10);
		}
		fprintf(fp,"\n");
	}	
	
	//free
	cudaFree(matrix_cuda); checkCudaError();
	checkCurandError(curandDestroyGenerator(generator));
	free(matrix);

	return 0;
}