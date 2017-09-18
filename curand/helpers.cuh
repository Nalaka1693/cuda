/* Header file for CUDA helper functions
Include this file in your code if you want to use the function checkCudaError() 
checkCudaError() function checks if the last cuda function call or kernel launch caused an error
and if yes it will print the error message and will abort the program

Also the function checkCublasError(errorcode) will check errors in CUBLAS library

These functions are based on the helper functions provided in cuda samples
*/

#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }
#define checkCublasError(errorcode) { checkCublas(errorcode,__FILE__, __LINE__); }
#define checkCurandError(errorcode) { checkCurand(errorcode,__FILE__, __LINE__); }

/* check whether the last CUDA function or CUDA kernel launch is erroneous and if yes an error message will be printed
and then the program will be aborted*/
void gpuAssert(const char *file, int line);

/*check whether a returned error code by a cublas api function is a erroneous and if yes print the error message*/
void checkCublas(int code,const char *file, int line);

/*Return the error message based the error code for cublas */
static const char *cublasGerErrorString(int error);

/*check whether a returned error code by a curand api function is a erroneous and if yes print the error message*/
void checkCurand(int code,const char *file, int line);

/*Return the error message based the error code for curand */
static const char *curandGerErrorString(int error);

/* Check whether a previous memory allocation was successful. If RAM is full usually the returned value is a NULL pointer.
For example if you allocate memory by doing 
int *mem = malloc(sizeof(int)*SIZE)
check whether it was successful by calling
isMemoryFull(mem) afterwards */
void checkAllocRAM(void *ptr);

/* This checks whether a file has been opened corrected. If a file opening failed the returned value is a NULL pointer
FOr example if you open a file using
FILE *file=fopen("file.txt","r");
check by calling isFileValid(file); */
void isFileValid(FILE *fp);
