/* C file that implements the helper functions specified in helpers.cu */

#include <stdio.h>
#include "helpers.cuh"
#include <cublas_v2.h>	
#include <curand.h>
	
/* check whether the last CUDA function or CUDA kernel launch is erroneous and if yes an error message will be printed
and then the program will be aborted*/

void gpuAssert(const char *file, int line){

	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "Cuda error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line );
        exit(-1);
   }
}

/*check whether a returned error code by a cublas api function is a erroneous and if yes print the error message*/
void checkCublas(int code,const char *file, int line){
	if(code!=CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Cublas error: %s \n in file : %s line number : %d", cublasGerErrorString(code), file, line );
		exit(-1);
	}

}

/*check whether a returned error code by a curand api function is a erroneous and if yes print the error message*/
void checkCurand(int code,const char *file, int line){
	if(code!=CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "Cublas error: %s \n in file : %s line number : %d", curandGerErrorString(code), file, line );
		exit(-1);
	}

}

/*Return the error message based the error code for cublas */
static const char *cublasGerErrorString(int error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

/*Return the error message based the error code for curand */
static const char *curandGerErrorString(int error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

/* Check whether a previous memory allocation was successful. If RAM is full usually the returned value is a NULL pointer.
For example if you allocate memory by doing 
int *mem = malloc(sizeof(int)*SIZE)
check whether it was successful by calling
checkAllocRAM(mem) afterwards */

void checkAllocRAM(void *ptr){
	if (ptr==NULL){
		fprintf(stderr, "Memory Full.\nYour array is too large. Please try a smaller array.\n");
		exit(EXIT_FAILURE);
	}
}

/* This checks whether a file has been opened corrected. If a file opening failed the returned value is a NULL pointer
FOr example if you open a file using
FILE *file=fopen("file.txt","r");
check by calling isFileValid(file); */

void isFileValid(FILE *fp){
	if (fp==NULL){
		perror("A file access error occurred\n");
		exit(EXIT_FAILURE);
	}
}