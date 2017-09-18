#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdio.h>
#include "helpers.cuh"

//mask width and height for blurring
#define MASKSIZE 15

#define INPUTFILE "image.jpg"
#define OUTPUTFILE "out.jpg"

using namespace cv;

//image blurring function in cuda
__global__ void blur(unsigned char *outImage,unsigned char *inImage,int rows,int cols){
	
	int i,j,k,m,z;
	
	//derive the row and column based on thread configuration
	i = blockIdx.y*blockDim.y + threadIdx.y;
	j = blockIdx.x*blockDim.x + threadIdx.x;
	z = blockIdx.z*blockDim.z + threadIdx.z;
	
	//Limit calculations for valid indices
	if(i < rows && j < cols){
				
		//average the color values of nearby pixels that falls in the mask to calculate the blurred pixel
				
		int sum = 0;
				
		//go through each pixel inside the mask
		for(k=i-MASKSIZE/2; k<i+MASKSIZE/2+1; k++){
			for(m=j-MASKSIZE/2; m<j+MASKSIZE/2+1; m++){
							
				//prevent accessing out of bound pixels
				if(k>=0 && k<rows && m>=0 && m<cols){
						//get the sum of  corresponding pixels
						sum += inImage[3*(k*cols+m) + z];
				}
			}
		}
				
		//colour value of output image's pixel
		outImage[3*(i*cols+j) + z]=(unsigned char)(sum/(MASKSIZE*MASKSIZE));

	}

}


int main( int argc, char** argv ){
	
	//space for input image
    Mat inImage; 
	// Read the input file	
    inImage = imread(INPUTFILE, CV_LOAD_IMAGE_COLOR);   	
	// Check for invalid input
    if(!inImage.data ){
        fprintf(stderr,"Could not open or find the image") ;
        return -1;
    }

	//space for output image
	Mat outImage(inImage.rows, inImage.cols, CV_8UC3, Scalar(0, 0, 0));	
	/*The 8U means the 8-bit Unsigned integer, C3 means 3 Channels for RGB color, and Scalar(0, 0, 0) is the initial value for each pixel. */


/********************************** CUDA stuff starts here *******************************/

	//start meauring time
	cudaEvent_t start,stop;
	float elapsedtime;
	cudaEventCreate(&start);
	cudaEventRecord(start,0);	
	
	//pointers for memory allocation in cudaa
	unsigned char *d_in;
	unsigned char *d_out;
	
	//allocate memory in cuda
	cudaMalloc((void **)&d_in,  sizeof(unsigned char)*inImage.rows*inImage.cols*3 ); checkCudaError();
	cudaMalloc((void **)&d_out, sizeof(unsigned char)*inImage.rows*inImage.cols*3 ); checkCudaError();

	//copy memory from ram to cuda
	cudaMemcpy(d_in, inImage.data, sizeof(unsigned char)*inImage.rows*inImage.cols*3, cudaMemcpyHostToDevice ); checkCudaError();
	
	//multiply the matrices in cuda
	dim3 threadsPerBlock(16,16,1);
	dim3 numBlocks(ceil(inImage.cols/(float)16),ceil(inImage.rows/(float)16), 3);
	blur<<<numBlocks,threadsPerBlock>>>(d_out,d_in,inImage.rows,inImage.cols);
	/*kernel calls are asynchronous. Hence the checkCudaError() function will execute before the kernel finished. 
	In order to tell wait till the kernel is over we use cudaDeviceSynchronize() before checkCudaError()
	In previous error checking examples it should be corrected as this*/
	cudaDeviceSynchronize(); checkCudaError();
	
	//copy the answer back from cuda to ram
	cudaMemcpy(outImage.data, d_out, sizeof(unsigned char)*inImage.rows*inImage.cols*3, cudaMemcpyDeviceToHost ); checkCudaError();

	//free the cuda memory
	cudaFree(d_in); checkCudaError();
	cudaFree(d_out); checkCudaError();
	
	//end measuring time
	cudaEventCreate(&stop);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedtime,start,stop);
	
	/********************** CUDA stuff ends here ********************************/

	
	//write the output image
    imwrite(OUTPUTFILE, outImage );
	
	//calculate the time taken and print to stderr
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 	
	
	
    return 0;
}