/************************************ The first CUDA C program ******************************************************************
* This is actually not CUDA C but pure C                                                                                     *
* All the operations here including printf works on the CPU                                                             *
* But the objective is to show that all general C syntax works for CUDA C as well and those general C statements happen on CPU  *
* The extension for CUDA C programs must be .cu                                                                              *
* You can compile using the nvidia C compiler by running the command  nvcc simpleprintf.cu -o firstprogram           *
* Then simply execute by running the command ./firstprogram                                                             *
*********************************************************************************************************************************/

#include <stdio.h>

int main() {
        int a = 1;
        int b = 3;
        int c = a + b * a - b;
        printf("Hello!\n");
        printf("answer is %d\n",c);

        return 0;
}
