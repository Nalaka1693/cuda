#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <stdio.h>

#define INPUTFILE "image.jpg"
#define OUTPUTFILE "out.jpg"

using namespace cv;

//grey scale conversion
void greyscale(unsigned char *outImage,unsigned char *inImage,int rows,int cols){
	int i,j;
	
	//go through each row
	for(i=0;i<rows;i++){
		//go through each column
		for(j=0;j<cols;j++){
				
				//get color values
				unsigned char blue=inImage[3*(i*cols+j)];
				unsigned char green=inImage[3*(i*cols+j)+1];
				unsigned char red=inImage[3*(i*cols+j)+2];
				
				//convert and save
				float grey=0.114*blue + 0.587*green + 0.299*red;
				outImage[i*cols+j]=(unsigned char)grey;

		}
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
	
	clock_t start = clock();
	//call blurring function
	greyscale(outImage.data,inImage.data,inImage.rows,inImage.cols);
	clock_t stop = clock();
	
	//write the output image
    imwrite(OUTPUTFILE, outImage);
	
	//calculate the time taken and print to stderr
	double elapsedtime = (stop-start)/(double)CLOCKS_PER_SEC;
	fprintf(stderr,"Elapsed time for operation on CPU is %1.5f seconds \n",elapsedtime);	
	
    return 0;
}