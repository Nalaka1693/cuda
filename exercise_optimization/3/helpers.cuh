/* Header file for CUDA helper functions
Include this file in your code if you want to use the function checkCuda() 
checkCuda() function checks if the value returned by a cuda function call is referrring to and error
and if yes it will print the correct error message and will abort the program
*/

/* get the error message based on the return value of a called function*/
static const char *getcudaError(cudaError_t error);

/* check whether the returned value of a function call is errorneous and if yes an error message will be printed
and then the program will be aborted*/
void checkCuda(cudaError_t status);

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
