#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdio.h>
#include "helpers.cuh"

#define INPUTFILE "image.jpg"
#define OUTPUTFILE "out.jpg"

using namespace cv;

//grey scale conversion
__global__ void grey(unsigned char *outImage,unsigned char *inImage,int rows,int cols){
	
	int i,j;
	
	//derive the row and column based on thread configuration
	i = blockIdx.y*blockDim.y + threadIdx.y;
	j = blockIdx.x*blockDim.x + threadIdx.x;
	
	//Limit calculations for valid indices
	if(i < rows && j < cols){
				
			//get color values
			unsigned char blue=inImage[3*(i*cols+j)];
			unsigned char green=inImage[3*(i*cols+j)+1];
			unsigned char red=inImage[3*(i*cols+j)+2];
				
			//convert and save
			float grey=0.114*blue + 0.587*green + 0.299*red;
			outImage[i*cols+j]=(unsigned char)grey;

	
	}
}


int main(){
	
	//space for input image
    Mat inImage; 
	// Read the input file	
    inImage = imread(INPUTFILE, CV_LOAD_IMAGE_COLOR);   	
	// Check for invalid input
    if(!inImage.data ){
        fprintf(stderr,"Could not open or find the image");
        return -1;
    }

	//space for output image
	Mat outImage(inImage.rows, inImage.cols, CV_8UC1, Scalar(0));	
	/*The 8U means the 8-bit Unsigned integer, C1 means 1 Channel for grey color, and Scalar(0) is the initial value for each pixel. */
	
	
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
	cudaMalloc((void **)&d_out, sizeof(unsigned char)*inImage.rows*inImage.cols ); checkCudaError();

	//copy memory from ram to cuda
	cudaMemcpy(d_in, inImage.data, sizeof(unsigned char)*inImage.rows*inImage.cols*3, cudaMemcpyHostToDevice ); checkCudaError();
	
	//start meauring time
	cudaEvent_t start2,stop2;
	float elapsedtime2;
	cudaEventCreate(&start2);
	cudaEventRecord(start2,0);
	
	//multiply the matrices in cuda
	dim3 threadsPerBlock(16,16);
	dim3 numBlocks(ceil(inImage.cols/(float)16),ceil(inImage.rows/(float)16));
	grey<<<numBlocks,threadsPerBlock>>>(d_out,d_in,inImage.rows,inImage.cols);
	cudaDeviceSynchronize(); checkCudaError();
	
	//end measuring time
	cudaEventCreate(&stop2);
	cudaEventRecord(stop2,0);
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&elapsedtime2,start2,stop2);
	
	//copy the answer back from cuda to ram
	cudaMemcpy(outImage.data, d_out, sizeof(unsigned char)*inImage.rows*inImage.cols, cudaMemcpyDeviceToHost ); checkCudaError();

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
    imwrite(OUTPUTFILE, outImage);
	
	//calculate the time taken and print to stderr
	fprintf(stderr,"Time spent for operation on CUDA(Calculation only) is %1.5f seconds\n",elapsedtime2/(float)1000); 	
	fprintf(stderr,"Time spent for operation on CUDA(Including memory allocation and copying) is %1.5f seconds\n",elapsedtime/(float)1000); 	
		
	
    return 0;
}