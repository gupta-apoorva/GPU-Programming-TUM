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
using namespace std;


// uncomment to use the camera
//#define CAMERA

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


__global__ void gradient_rand(float* imgOut_cuda,float* cuda_v1,float* cuda_v2,int w,int h,int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    if (t_numx < w && t_numy < h && t_numz < nc){
        int x_neg = t_numx-1;
        int x_pos = t_numx+1;
        int y_neg = t_numy-1;
        int y_pos = t_numy+1;
        if (x_neg<0) x_neg = 0;
        if (x_pos>=w) x_pos = w-1;
        if (y_neg<0) y_neg = 0;
        if (y_pos>=h) y_pos = h-1;

        cuda_v1[t_numx + w*t_numy + w*h*t_numz] = 1.f/32.f*(    3*imgOut_cuda[x_pos + w*y_pos + w*h*t_numz] 
                                                            +   10*imgOut_cuda[x_pos + w*t_numy + w*h*t_numz] 
                                                            +   3*imgOut_cuda[x_pos + w*y_neg + w*h*t_numz] 
                                                            -   3*imgOut_cuda[x_neg + w*y_pos + w*h*t_numz]
                                                            -   10*imgOut_cuda[x_neg + w*t_numy + w*h*t_numz] 
                                                            -   3*imgOut_cuda[x_neg + w*y_neg + w*h*t_numz]) ;

        cuda_v2[t_numx + w*t_numy + w*h*t_numz] = 1.f/32.f*(    3*imgOut_cuda[x_pos + w*y_pos + w*h*t_numz] 
                                                            +   10*imgOut_cuda[t_numx + w*y_pos + w*h*t_numz] 
                                                            +   3*imgOut_cuda[x_neg + w*y_pos + w*h*t_numz] 
                                                            -   3*imgOut_cuda[x_pos + w*y_neg + w*h*t_numz]
                                                            -   10*imgOut_cuda[t_numx + w*y_neg + w*h*t_numz] 
                                                            -   3*imgOut_cuda[x_neg + w*y_neg + w*h*t_numz]) ;
    }
}

__global__ void cuda_convolution(float *cuda_imgIn, float* imgOut_cuda, float* kernel_cuda, int w, int h, int nc, int radius){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    float output = 0;
    if (t_numx < w && t_numy < h && t_numz < nc){
        int index = t_numx + w*t_numy + w*h*t_numz;
        //imgOut_cuda[index] = 0;
        for (int p = -radius; p <= radius; p++) {
            for (int q =-radius; q <= radius; q++) {
                int temp_j = t_numy + p;
                int temp_i = t_numx + q;
                if (temp_i<0) temp_i = 0;
                if (temp_i>=w) temp_i = w-1;
                if (temp_j<0) temp_j = 0;
                if (temp_j>=h) temp_j = h-1;
                int image_index = temp_i + temp_j*w + t_numz*h*w;
                int kernel_index = q+radius + (2*radius+1)*(p+radius);
                 output += cuda_imgIn[image_index] * kernel_cuda[kernel_index];
            }
        }
    imgOut_cuda[index] = output;
    }    
}

__global__ void M_calculation(float *M_out_cuda, float* cuda_v1, float* cuda_v2, int w, int h, int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    int index = t_numx + w*t_numy + w*h*t_numz;
    if (t_numx < w && t_numy < h && t_numz < nc){
        M_out_cuda[t_numx + w*t_numy + w*h*0] += cuda_v1[index]*cuda_v1[index];
        M_out_cuda[t_numx + w*t_numy + w*h*1] += cuda_v1[index]*cuda_v2[index];
        M_out_cuda[t_numx + w*t_numy + w*h*2] += cuda_v2[index]*cuda_v2[index];
    }

}


__device__ void eigenvalues_eigenvectors (float m11,float m12,float m22, float* lambda1,float* lambda2, float* vector1, float* vector2){
    float T = m11 + m22;
    float D =  m11*m22 - m12*m12;

    *lambda1 = T/2.f + sqrt(pow(T,2.f)/4.f - D);
    *lambda2 = T/2.f - sqrt(pow(T,2.f)/4.f - D);

    if (m12 ==0){
        vector1[0] = 1;
    vector1[1] = 0;
    vector2[0] = 0;
    vector2[1] = 1;
    
    }

    /*else if ((m11-*lambda1)/m12 == m12/(m22-*lambda1)){
        vector1[0] = -m12/(m11-*lambda1);
        vector1[1] = 1;  
    }
    else if ((m11-*lambda2)/m12 == m12/(m22-*lambda2)){
        vector2[0] = -m12/(m11-*lambda2);
        vector2[1] = 1;     
    }*/

}

__global__ void diffusion(float* cuda_v1, float* cuda_v2, float* G_out_cuda, int  w, int h, int nc, float alpha, float C){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    float lambda1;
    float lambda2;
    float vector1[2] = {0,0};
    float vector2[2] = {0,0};
    float m11, m12, m22;
    float mu_1 = alpha;
    float mu_2;
    float g[4];
    int index = t_numx + w*t_numy + w*h*t_numz;
    if (t_numx < w && t_numy < h && t_numz <nc){
       m11 = G_out_cuda[t_numx + w*t_numy + w*h*0];
       m12 = G_out_cuda[t_numx + w*t_numy + w*h*1];
       m22 = G_out_cuda[t_numx + w*t_numy + w*h*2];


        eigenvalues_eigenvectors(m11, m12, m22, &lambda1, &lambda2, vector1, vector2);
        if (lambda1 == lambda2){
            mu_2 = alpha;
        }
        else{
            mu_2 = alpha + (1 - alpha)*exp(-C/pow(lambda1 - lambda2, 2));
        }

        g[0] = mu_1*pow((float)vector1[0] ,(float)2) + mu_2*pow((float)vector2[0] ,(float)2);
        g[1] = mu_1*vector1[0]*vector1[1] + mu_2*vector2[0]*vector2[1]; 
        g[2] = mu_1*vector1[0]*vector1[1] + mu_2*vector2[0]*vector2[1];
        g[3] = mu_1*pow((float)vector1[1] ,(float)2) + mu_2*pow((float)vector2[1], (float)2); 

        cuda_v1[index] = g[0]*cuda_v1[index] + g[1]*cuda_v2[index];
        cuda_v2[index] = g[2]*cuda_v1[index] + g[3]*cuda_v2[index];
    }
}

__global__ void divergence_update(float* cuda_imgIn,float* cuda_div , float* cuda_v1, float* cuda_v2, int w, int h, int nc, float tau ){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    int index = t_numx + w*t_numy + w*h*t_numz;
    if (t_numx > 0 && t_numx < w && t_numy < h && t_numz < nc){
        cuda_div[index] +=  cuda_v1[index] - cuda_v1[index -1];
    }
__syncthreads();
    if (t_numy > 0 && t_numx < w && t_numy < h && t_numz < nc){
        cuda_div[index] += cuda_v2[index] - cuda_v2[index - w];
    }
__syncthreads();
    if (t_numx < w && t_numy < h && t_numz < nc){
        cuda_imgIn[index] += (float)tau*cuda_div[index]; 
    }
}


int main(int argc, char **argv)
{
    cudaDeviceSynchronize();  CUDA_CHECK;

#ifdef CAMERA
#else
    // input image
    string image = "";
    float sigma = 0.5;
    float rho = 3;
    int iterations = 10;
    float tau = 0.01;
    float alpha = 0.01;
    float C= 0.000005;

    bool ret = getParam("i", image, argc, argv);
        if (!ret) cerr << "ERROR: no image specified" << endl;

    bool ret2 = getParam("iter", iterations, argc, argv);
        if (!ret2) {cerr << "ERROR: Num of iterations not specified" << endl; return 1;}
        
    bool ret3 = getParam("C", C, argc, argv);
        if (!ret3) {cerr << "ERROR: no C specified" << endl; return 1;}
        
    bool ret4 = getParam("alpha", alpha, argc, argv);
        if (!ret4) {cerr << "ERROR: no alpha specified" << endl; return 1;}
       
    bool ret5 = getParam("sigma", sigma, argc, argv);
        if (!ret5) {cerr << "ERROR: no sigma specified" << endl; return 1;}

    bool ret6 = getParam("rho", rho, argc, argv);
        if (!ret6) {cerr << "ERROR: no rho specified" << endl; return 1;}

    bool ret7 = getParam("tau", tau, argc, argv);
        if (!ret7) {cerr << "ERROR: no tau specified" << endl; return 1;}
        
    if (argc <= 7) { cout << "Usage: " << argv[0] << " -i <image> -iter <iterations> -C <C> -alpha <alpha> -sigma <sigma> -rho <rho> -tau <tau> [-repeats <repeats>] [-gray]" << endl; return 1; }
    #endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
    cv::VideoCapture camera(0);
    if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
    camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;

    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);

/// Calculating 1st kernel
    int radius = ceil(3*sigma);
    int total_ele_filter = (int)pow(2*radius + 1, 2);
    float* kernel = new float[total_ele_filter];
    gaussian_kernel(kernel, sigma, radius);
    float *kernel_cuda;
    cudaMalloc((void**)&kernel_cuda, total_ele_filter*sizeof(float));
    cudaMemcpy(kernel_cuda, kernel, total_ele_filter*sizeof(float), cudaMemcpyHostToDevice);

///  calculating different kernel
    int radius_diff = ceil(3*rho);
    int total_ele_filter_diff = (int)pow(2*radius_diff + 1, 2);
    float* kernel_diff = new float[total_ele_filter_diff];
    gaussian_kernel(kernel_diff, rho, radius_diff);
    float *kernel_cuda_diff;
    cudaMalloc((void**)&kernel_cuda_diff, total_ele_filter_diff*sizeof(float));
    cudaMemcpy(kernel_cuda_diff, kernel_diff, total_ele_filter_diff*sizeof(float), cudaMemcpyHostToDevice);	

/// Allocating memory for image data and result in cuda
    int array_size = w*h*nc;
    float *cuda_imgIn, *imgOut_cuda;
    cudaMalloc((void**)&cuda_imgIn , array_size*sizeof(float));
    cudaMalloc((void**)&imgOut_cuda , array_size*sizeof(float));
    cudaMemcpy(cuda_imgIn, imgIn , array_size*sizeof(float) , cudaMemcpyHostToDevice); 

/// Defining the grid and block size. 
    dim3 block = dim3(32,32,1);
    int grid_x = ((w + block.x - 1)/block.x);
    int grid_y = ((h + block.y - 1)/block.y);
    int grid_z = ((nc + block.z - 1)/block.z);
    dim3 grid = dim3(grid_x, grid_y, grid_z );
   
/// Allocating memory for forward differencs calculation on cuda 
    float* cuda_v1;
    float* cuda_v2;
    cudaMalloc((void**) &cuda_v1, array_size*sizeof(float));
    cudaMalloc((void**) &cuda_v2, array_size*sizeof(float));

// Allocating memory for the M and G mmatrix on the cuda
    float *M_out_cuda, *G_out_cuda;
    cudaMalloc((void**)&M_out_cuda , array_size*sizeof(float));
    cudaMalloc((void**)&G_out_cuda , array_size*sizeof(float));
  
/// Allocating memory for the divergence on cuda
    float *cuda_div;
    cudaMalloc((void**) &cuda_div , array_size*sizeof(float));

    for (int iter=0; iter<iterations; iter++){

        cudaMemset(imgOut_cuda, 0 , array_size*sizeof(float));   
        cuda_convolution <<<grid, block>>> (cuda_imgIn, imgOut_cuda, kernel_cuda, w, h, nc, radius);

        cudaMemset(cuda_v1, 0 , array_size*sizeof(float));
        cudaMemset(cuda_v2, 0 , array_size*sizeof(float));

        gradient_rand <<<grid, block>>>(imgOut_cuda, cuda_v1, cuda_v2, w, h, nc );

        cudaMemset(M_out_cuda, 0 , array_size*sizeof(float));
        M_calculation <<< grid, block >>> (M_out_cuda, cuda_v1, cuda_v2, w, h, nc );

        cudaMemset(G_out_cuda, 0 , array_size*sizeof(float));  
        cuda_convolution <<< grid, block >>> (M_out_cuda, G_out_cuda, kernel_cuda_diff, w, h, nc, radius_diff);

        diffusion <<< grid, block >>> ( cuda_v1, cuda_v2, G_out_cuda, w, h, nc, alpha, C);
        //cudaMemcpy(imgOut, cuda_v1 , w*h*nc*sizeof(float) , cudaMemcpyDeviceToHost);

        cudaMemset(cuda_div, 0, array_size*sizeof(float));
        divergence_update <<< grid, block >>> (cuda_imgIn, cuda_div , cuda_v1, cuda_v2, w, h, nc, tau );
}
    
    cudaMemcpy(imgOut, cuda_imgIn , array_size*sizeof(float) , cudaMemcpyDeviceToHost);

    convert_layered_to_mat(mOut,imgOut);
    showImage("OUTPUT", mOut, 0, 0);
    showImage("Input", mIn, 100, 100); 
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

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] kernel;
    delete[] kernel_diff;

    cudaFree(cuda_imgIn);
    cudaFree(cuda_v2);
    cudaFree(cuda_v1);
    cudaFree(cuda_div);
    cudaFree(G_out_cuda);
    cudaFree(M_out_cuda);
    cudaFree(kernel_cuda);
    cudaFree(kernel_cuda_diff);
    cudaFree(imgOut_cuda);

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



