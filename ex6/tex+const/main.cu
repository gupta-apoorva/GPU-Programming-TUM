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
#include "helper.h"
#include <math.h>
using namespace std;

# define r_max 20
__constant__ float kernel_cuda[(2*r_max + 1)*(2*r_max + 1)];

texture<float,2,cudaReadModeElementType> texRef;

__host__ void gaussian_kernel(float *kernel, float sigma, int radius){
    float sum = 0 ;
    for(int j = -radius; j<=radius ; j++){
            for (int i=-radius; i<=radius; i++){
                int index = i+radius + (2*radius+1)*(j+radius);
                kernel[index] = 0.5/3.14159/pow(sigma,2.0)*pow(2.71828,-(pow(i,2) + pow(j,2))/2/(pow(sigma,2)));  
                sum = sum +  kernel[index];   
        }
    }

    for (int i=0; i<(2*radius + 1)*(2*radius + 1); i++){
        kernel[i] = kernel[i] / sum;
    }
}

__host__ void scaled_gaussian_kernel(float* kernel_out, const float *kernel , int total_ele_filter){
    float max_value = 0;
    for(int i=0; i<total_ele_filter; i++){
        if (max_value < kernel[i]){
            max_value = kernel[i];      
        }
    }
    
    float scaled_value = 1/max_value;
    for(int i=0; i<total_ele_filter; i++){
        kernel_out[i] = scaled_value*kernel[i]*255.f;   
    }
}

__host__ void visualize_kernel(float* kernel_out, int radius){
    cv::Mat kernel_image(2*radius + 1,2*radius + 1, CV_32FC1);
    convert_layered_to_mat(kernel_image, (float*)kernel_out);  
    showImage("Kernel Output", kernel_image, 2*radius +1, 2*radius +1);
}

__host__ void convolution(float *imgIn, float* imgOut, float* kernel, int w, int h, int nc, int radius){
    for (int z=0; z<nc; z++){
        for (int y=0; y<h; y++){
            for (int x=0; x<w; x++){
                int index = x + y*w + z*h*w;
                for (int p = -radius; p <= radius; p++) {
                    for (int q =-radius; q <= radius; q++) {
                        int temp_j = y + p;
                        int temp_i = x + q;
                        if (temp_i<0) temp_i = 0;
                        if (temp_i>=w) temp_i = w-1;
                        if (temp_j<0) temp_j = 0;
                        if (temp_j>=h) temp_j = h-1;
                        int indexOffset = temp_i + temp_j*w + z*h*w;
                        int kernel_index = q+radius + (2*radius+1)*(p+radius);
                        imgOut[index] += imgIn[indexOffset] * kernel[kernel_index];
                    }
                }
                
            }
        }
    }
}


__global__ void cuda_convolution_texture(float* imgOut_cuda, int w, int h, int nc, int radius){

    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    int index = t_numx + w*t_numy + w*h*t_numz;
    if (t_numx < w && t_numy < h && t_numz < nc)
        for (int p = -radius; p <= radius; p++) {
            for (int q =-radius; q <= radius; q++) {
                int kernel_index = q+radius + (2*radius+1)*(p+radius);
                imgOut_cuda[index] += tex2D(texRef , t_numx + q , t_numy + p + t_numz*h) * kernel_cuda[kernel_index];
            }
        }
    }    


__global__ void copy_kernel(float* output_cuda){
    output_cuda[threadIdx.x] = kernel_cuda[threadIdx.x];
}

int main(int argc, char **argv){
       cudaDeviceSynchronize();  CUDA_CHECK;


    // input image
    string image = "";
    float sigma = 3;
    bool ret = getParam("i", image, argc, argv);
    bool ret2 = getParam("sigma", sigma, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (!ret2) {cerr << "ERROR: no sigma specified" << endl; return 1;}
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -sigma <sigma>[-repeats <repeats>] [-gray]" << endl; return 1; }

    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    


    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];


   convert_mat_to_layered (imgIn, mIn);


// Calculating Gaussian kernel
    // Provided Sigma
    //float sigma = 3; //Needed to maintain the r_max

    int radius = ceil(3*sigma);

    int total_ele_filter = (int)pow(2*radius + 1, 2);
    float* kernel = new float[total_ele_filter];

    gaussian_kernel(kernel, sigma, radius);


    //float* kernel_out = new float[total_ele_filter];

    //scaled_gaussian_kernel(kernel_out, kernel, total_ele_filter);
    
    //visualize_kernel(kernel_out, radius);

// CPU convolution 
    
    //convolution(imgIn, imgOut, kernel, w, h, nc, radius);

    cudaMemcpyToSymbol(kernel_cuda, kernel , total_ele_filter*sizeof(float));

/*float* output_cuda;
cudaMalloc((void**)&output_cuda, total_ele_filter*sizeof(float));
cudaMemset(output_cuda, 0 , total_ele_filter*sizeof(float));

copy_kernel <<< 1, total_ele_filter >>> (output_cuda);

float* output = new float[total_ele_filter];

cudaMemcpy(output, output_cuda, total_ele_filter*sizeof(float), cudaMemcpyDeviceToHost);

for (int i = 0; i<total_ele_filter ; i++){
    cout << output[i] << endl;
}*/


// GPU convolution with texture

    texRef.addressMode[0] = cudaAddressModeClamp; // clamp x to border
    texRef.addressMode[1] = cudaAddressModeClamp; // clamp y to border
    texRef.filterMode = cudaFilterModePoint; // linear interpolation
    texRef.normalized = false;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    float *image_cuda;
    size_t pitch;
    cudaMallocPitch((void**)&image_cuda, &pitch, w*sizeof(float), h*nc);
    cudaMemcpy2D(image_cuda, pitch, imgIn, w*sizeof(float), w*sizeof(float), h*nc, cudaMemcpyHostToDevice);
    cudaBindTexture2D(NULL, &texRef, image_cuda, &desc, w, h*nc, pitch);
    
    float *imgOut_cuda;
    cudaMalloc((void**)&imgOut_cuda , h*w*nc*sizeof(float));
    cudaMemset(imgOut_cuda , 0, h*w*nc*sizeof(float));
    //cudaMemcpy(imgOut_cuda, imgOut , h*w*nc*sizeof(float) , cudaMemcpyHostToDevice);

    dim3 block = dim3(32,32,1);
    int grid_x = ((w + block.x - 1)/block.x);
    int grid_y = ((h + block.y - 1)/block.y);
    int grid_z = ((nc + block.z - 1)/block.z);
    dim3 grid = dim3(grid_x, grid_y, grid_z );

    Timer timer; timer.start();
    cuda_convolution_texture <<<grid, block>>> (imgOut_cuda, w, h, nc, radius);
     timer.end();  float t = timer.get();  // elapsed time in seconds
    cout << "gpu time: " << t*1000 << " ms" << endl;
    cudaMemcpy(imgOut, imgOut_cuda, h*w*nc*sizeof(float), cudaMemcpyDeviceToHost);
    cudaUnbindTexture(&texRef);

    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif


    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    delete[] imgIn;
    delete[] imgOut;
    delete[] kernel;
    cudaFree(image_cuda);
    cudaFree(imgOut_cuda);

    // close all opencv windows
    cvDestroyAllWindows();
    


 
    return 0;
}



