// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include "helper.h"
#include <iostream>
#include <math.h>
#include "cublas.h"
#include <cuda_runtime.h>
using namespace std;


// uncomment to use the camera
//#define CAMERA


__global__ void histogram256_global(float* img, int* histo, int w, int h, int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    float value = 0;
    if (t_numx + w*t_numy + t_numz  < w*h){
       for (int j =0; j < nc; j++){
            value += img[t_numx + w*t_numy + j*t_numz*w*h]; 
        }
	value = value*255.f/float(nc);        
        atomicAdd((int*)&(histo[(int)value]), 1);  
    }
}

__global__ void histogram256_shared(float* img, int* histo, int w, int h, int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;


    float value = 0;
    __shared__ int block_histo[256];
     if ( threadIdx.x == 0 && threadIdx.y == 0 ){
	for(int i=0 ; i<256 ; i++)
        	block_histo[i] = 0;
    }

__syncthreads();
    if (t_numx + w*t_numy + t_numz  < w*h){
        for (int j =0; j < nc; j++){
            value += img[t_numx + w*t_numy + j*t_numz*w*h]; 
        }
	value = value*255.f/float(nc);
        atomicAdd((int*)&(block_histo[(int)value]), 1);  
    }

__syncthreads();
    if ( threadIdx.x == 0 && threadIdx.y == 0 ){
	for(int i=0 ; i<256 ; i++)
            atomicAdd((int*)&(histo[i]), block_histo[i]);
}
__syncthreads();
}


int main(int argc, char **argv)
{
    cudaDeviceSynchronize();  CUDA_CHECK;   

    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }

    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
    mIn.convertTo(mIn,CV_32F);
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << " x " << nc <<endl;


    float *imgIn = new float[(size_t)w*h*nc];

    convert_mat_to_layered (imgIn, mIn);

    
    float* cuda_imgIn;
    int* cuda_histo256;
    cudaMalloc((void**) &cuda_imgIn , w*h*nc*sizeof(float));
    cudaMalloc((void**) &cuda_histo256 , 256*sizeof(int));
    cudaMemset(cuda_histo256 , 0, 256*sizeof(int));
    cudaMemcpy(cuda_imgIn, imgIn , w*h*nc*sizeof(float) , cudaMemcpyHostToDevice);

    dim3 block = dim3(32,32,1);
    int grid_x = ((w + block.x + 1)/block.x);
    int grid_y = ((h + block.y + 1)/block.y);
    int grid_z = 1;
    dim3 grid = dim3(grid_x, grid_y, grid_z );

    Timer timer;
    timer.start();
    histogram256_global <<<grid, block>>>(cuda_imgIn, cuda_histo256 , w, h, nc);
    timer.end();  
    float t_global = timer.get();  // elapsed time in seconds
    cout << " " << endl;
    cout << "time when using global atomics: " << t_global*1000 << " ms" << endl;
    cout << " " << endl;

    int* histo_global = new int[256];
    cudaMemcpy(histo_global, cuda_histo256, 256*sizeof(int), cudaMemcpyDeviceToHost);
    showHistogram256("HISTOGRAM_GLOBAL" , histo_global, 100 + w, 100); 

    cudaMemset(cuda_histo256 , 0, 256*sizeof(int));
    timer.start();
    histogram256_shared <<<grid, block>>>(cuda_imgIn, cuda_histo256 , w, h, nc);
    timer.end();  
    float t_shared = timer.get();  // elapsed time in seconds
    cout << "time when using shared atomics: " << t_shared*1000 << " ms" << endl;

    int* histo_shared = new int[256];
    cudaMemcpy(histo_shared, cuda_histo256, 256*sizeof(int), cudaMemcpyDeviceToHost);
    showHistogram256("HISTOGRAM_SHARED" , histo_shared, 100 + 2*w, 100); 

    cout << " " << endl;
    cout << "Percentage improvement with shared atomics: "<< 100*((t_global - t_shared)/t_shared) << endl;
    cout << " " << endl;


    // show input image
    showImage("Input", mIn, 100, 100);  

    cv::waitKey(0);

    // free allocated arrays
    delete[] imgIn;
    delete[] histo_shared;
    delete[] histo_global;
    cudaFree(cuda_imgIn);
    cudaFree(cuda_histo256);


    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



