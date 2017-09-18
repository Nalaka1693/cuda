#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define WIDTH 2048
#define HEIGHT 1536
#define XMIN -2.0
#define XMAX 1
#define YMIN -1.25
#define YMAX 1.25
#define INF 4
#define MAXN 3000
#define max(a,b) (((a)>(b))?(a):(b))

unsigned char image_cpu[WIDTH *HEIGHT * 3];

__device__ unsigned char image_cuda[WIDTH *HEIGHT * 3];

/**********************************************************************************/

/*Calculate R value in RGB based on divergence*/
__device__ unsigned char red(int i){
        if (i==0 )
           return 0 ;
        else
        return ((i+10)%256);
}

/*Calculate B value in RGB based on divergence*/
__device__ unsigned char blue(int i){
        if (i==0)
        return  0;
        else
        return ((i + 234) % 7 * (255/7));
}

/*Calculate G value in RGB based on divergence*/
__device__ unsigned char green(int i){
        if (i==0)
           return  0 ;
        else
           return ((i+100) % 9 * (255/9));
}

/***********************************************************************************/
/*Set calculations*/
//Transform a pixel to complex plane
__device__ float transform_to_x(int x){
        return XMIN+x*(XMAX-(XMIN))/(float)WIDTH;
}
__device__ float transform_to_y(int y){
        return YMAX-y*(YMAX-(YMIN))/(float)HEIGHT;
}

//check whether is in Mandelbrot set
__device__ int isin_mandelbrot(float realc,float imagc){
        int i=0;
        float realz_next=0,imagz_next=0;
        float abs=0;
        float realz=0;
        float imagz=0;
        while(i<MAXN && abs<INF){
           realz_next=realz*realz-imagz*imagz+realc;
           imagz_next=2*realz*imagz+imagc;
           abs=realz*realz+imagz*imagz;
           realz=realz_next;
           imagz=imagz_next;
           i++;
        }

        if (i==MAXN)
           return 0;
        else
           return i;
}

/************************************************************************************/
/*Creating the Mandelbrot image*/

__global__ void mandelbrot_image(){
        int x,y;
        int n;
        int mandel;

        x = blockIdx.y * blockDim.y + threadIdx.y;
        y = blockIdx.x * blockDim.x + threadIdx.x;
        n = 3 * (y * WIDTH + x);

        //calculate whether is in Mandelbrot set or otherwise the divergence. Based on that calculat ethe color
        if (x < HEIGHT && y < WIDTH) {
           mandel=isin_mandelbrot(transform_to_x(x),transform_to_y(y));
           image[n]=red(mandel);
           image[n+1]=green(mandel);
           image[n+2]=blue(mandel);
           n=n+3;
        }
}

/********************************************************************************************/

int main(int argc, char** argv) {

  //time calculations
  clock_t begin,end;
  begin=clock();

  dim3 blockNum(ceil(COLS2/(float)16), ceil(ROWS1/(float)16));
  dim3 threadsPerBlocks(16, 16);

  //actual work
  mandelbrot_image<<<blockNum, threadsPerBlocks>>>();

  cudaMemcpyFromSymbol(image_cpu, image_cuda);

  //finish time measurements
  end=clock();
  double cputime=(double)((end-begin)/(float)CLOCKS_PER_SEC);
  printf("Time using CPU for calculation is %.10f\n",cputime);

  //write to file
  // color component ( R or G or B) is coded from 0 to 255
  // it is 24 bit color RGB file
  const int MaxColorComponentValue=255;
  FILE * fp;
  char *filename="image.ppm";
  char *comment="# ";//comment should start with #

  //create new file,give it a name and open it in binary mode
  fp= fopen(filename,"wb"); // b -  binary mode
  //write ASCII header to the file
  fprintf(fp,"P6\n %s\n %d\n %d\n %d\n",comment,WIDTH,HEIGHT,MaxColorComponentValue);
  // compute and write image data bytes to the file
  fwrite(image,1,WIDTH *HEIGHT * 3,fp);
  //close the file
  fclose(fp);

  return 0;

}
