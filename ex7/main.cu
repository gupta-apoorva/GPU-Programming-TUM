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

__device__ void eigenvalues (float m11,float m12,float m22, float* lambda){
    float T = m11 + m22;
    float D =  m11*m22 - m12*m12;

    lambda[1] = (float)T/2.f + sqrt(pow((float)T,2.f)/4.f - (float)D);
    lambda[2] = (float)T/2.f - sqrt(pow((float)T,2.f)/4.f - (float)D);
}

__global__ void T_eigenvalues(float* imgIn_cuda, float* imgOut_cuda,float* T_out_cuda, int  w, int h, int nc, float alpha, float beta){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = 0;

    float *lambda = new float[2];
    float m11, m12, m22;
    if (t_numx < w && t_numy < h && t_numz < nc){
        m11 = T_out_cuda[t_numx + w*t_numy + w*h*0];
        m12 = T_out_cuda[t_numx + w*t_numy + w*h*1];
        m22 = T_out_cuda[t_numx + w*t_numy + w*h*2];

        eigenvalues(m11, m12, m22, lambda);

        if (lambda[2] >= lambda[1] && lambda[1] >= alpha){
            imgOut_cuda[t_numx + w*t_numy + w*h*0] = 255;
            imgOut_cuda[t_numx + w*t_numy + w*h*1] = 0;
            imgOut_cuda[t_numx + w*t_numy + w*h*2] = 0;
        }
        else if (lambda[1] >= alpha && alpha >=beta && beta >= lambda[2]){
            imgOut_cuda[t_numx + w*t_numy + w*h*0] = 255;
            imgOut_cuda[t_numx + w*t_numy + w*h*1] = 255;
            imgOut_cuda[t_numx + w*t_numy + w*h*2] = 0;
        }
        else{
            imgOut_cuda[t_numx + w*t_numy + w*h*0] = 0.5*imgIn_cuda[t_numx + w*t_numy + w*h*0];
            imgOut_cuda[t_numx + w*t_numy + w*h*1] = 0.5*imgIn_cuda[t_numx + w*t_numy + w*h*1];
            imgOut_cuda[t_numx + w*t_numy + w*h*2] = 0.5*imgIn_cuda[t_numx + w*t_numy + w*h*2];
        }
    }
}


__global__ void gradient_fd(float* imgOut_cuda,float* cuda_v1,float* cuda_v2,int w,int h,int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    if (t_numx + 1 < w && t_numy < h && t_numz < nc){
        cuda_v1[t_numx + w*t_numy + w*h*t_numz] = imgOut_cuda[t_numx + 1 + w*t_numy + w*h*t_numz] - imgOut_cuda[t_numx + w*t_numy + w*h*t_numz];
    }

    if (t_numx < w && t_numy + 1< h && t_numz < nc){
        cuda_v2[t_numx + w*t_numy + w*h*t_numz] = imgOut_cuda[t_numx + w*(t_numy+1) + w*h*t_numz] - imgOut_cuda[t_numx + w*t_numy + w*h*t_numz];
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

        cuda_v1[t_numx + w*t_numy + w*h*t_numz] = 1.f/32.f*(3*imgOut_cuda[x_pos + w*y_pos + w*h*t_numz] 
                                        + 10*imgOut_cuda[x_pos + w*t_numy + w*h*t_numz] 
                                        + 3*imgOut_cuda[x_pos + w*y_neg + w*h*t_numz] 
                                        - 3*imgOut_cuda[x_neg + w*y_pos + w*h*t_numz]
                                        - 10*imgOut_cuda[x_neg + w*t_numy + w*h*t_numz] 
                                        - 3*imgOut_cuda[x_neg + w*y_neg + w*h*t_numz]) ;

        cuda_v2[t_numx + w*t_numy + w*h*t_numz] = 1.f/32.f*( 3*imgOut_cuda[x_pos + w*y_pos + w*h*t_numz] 
                                        + 10*imgOut_cuda[t_numx + w*y_pos + w*h*t_numz] 
                                        + 3*imgOut_cuda[x_neg + w*y_pos + w*h*t_numz] 
                                        - 3*imgOut_cuda[x_pos + w*y_neg + w*h*t_numz]
                                        - 10*imgOut_cuda[t_numx + w*y_neg + w*h*t_numz] 
                                        - 3*imgOut_cuda[x_neg + w*y_neg + w*h*t_numz]) ;
    }
}


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

__global__ void cuda_convolution(float *imgIn_cuda, float* imgOut_cuda, float* kernel_cuda, int w, int h, int nc, int radius){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

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
                imgOut_cuda[index] += imgIn_cuda[image_index] * kernel_cuda[kernel_index];
            }
        }
    }    
}

__global__ void M_calculation(float *M_out_cuda, float* cuda_v1, float* cuda_v2, int w, int h, int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    int index = t_numx + w*t_numy + w*h*t_numz;
    if (t_numx < w && t_numy < h && t_numz < nc){
        M_out_cuda[t_numx + w*t_numy + w*h*t_numz] += cuda_v1[index]*cuda_v1[index];
        M_out_cuda[t_numx + w*t_numy + w*h*t_numz] += cuda_v1[index]*cuda_v2[index];
        M_out_cuda[t_numx + w*t_numy + w*h*t_numz] += cuda_v2[index]*cuda_v2[index];
    }
}



int main(int argc, char **argv)
{

    cudaDeviceSynchronize();  CUDA_CHECK;

#ifdef CAMERA
#else
    // input image
    string image = "";
    float sigma = 1;
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    bool ret2 = getParam("sigma", sigma, argc, argv);
    if (!ret2) cerr << "ERROR: no sigma specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> -sigma <sigma> [-repeats <repeats>] [-gray]" << endl; return 1; }
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

    convert_mat_to_layered (imgIn, mIn);

// CONVOLUTION ON GPU......START

    int radius = ceil(3*sigma);

    int total_ele_filter = (int)pow(2*radius + 1, 2);
    float* kernel = new float[total_ele_filter];

    gaussian_kernel(kernel, sigma, radius);


    float *kernel_cuda, *imgIn_cuda, *imgOut_cuda;
    cudaMalloc((void**)&kernel_cuda, total_ele_filter*sizeof(float));
    cudaMemcpy(kernel_cuda, kernel, total_ele_filter*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&imgIn_cuda , h*w*nc*sizeof(float));
    cudaMalloc((void**)&imgOut_cuda , h*w*nc*sizeof(float));
    cudaMemset(&imgOut , 0, h*w*nc*sizeof(float));
    cudaMemcpy(imgIn_cuda, imgIn , h*w*nc*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(imgOut_cuda, imgOut , h*w*nc*sizeof(float) , cudaMemcpyHostToDevice);

    dim3 block = dim3(32,32,1);
    int grid_x = ((w + block.x - 1)/block.x);
    int grid_y = ((h + block.y - 1)/block.y);
    int grid_z = ((nc + block.z - 1)/block.z);
    dim3 grid = dim3(grid_x, grid_y, grid_z );

    cuda_convolution <<<grid, block>>> (imgIn_cuda, imgOut_cuda, kernel_cuda, w, h, nc, radius);
//cudaMemcpy(imgOut, imgOut_cuda , w*h*nc*sizeof(float) , cudaMemcpyDeviceToHost);
 
// CONVOLUTION ON GPU......END

// GRADIENT CALCULATION START
    int array_size = w*h*nc;

    float* cuda_v1;
    float* cuda_v2;

    cudaMalloc((void**) &cuda_v1, array_size*sizeof(float));
    cudaMalloc((void**) &cuda_v2, array_size*sizeof(float));

    cudaMemcpy(cuda_v1, imgOut, array_size*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_v2, imgOut , array_size*sizeof(float) , cudaMemcpyHostToDevice);


    gradient_rand <<<grid, block>>>(imgOut_cuda, cuda_v1, cuda_v2, w, h, nc );

    //cudaMemcpy(imgOut, cuda_v1 , w*h*nc*sizeof(float) , cudaMemcpyDeviceToHost);

// GRADIENT CALCULATION END

    float *M_out_cuda;

    cudaMalloc((void**)&M_out_cuda , h*w*nc*sizeof(float));
    cudaMemcpy(M_out_cuda, imgOut , h*w*nc*sizeof(float) , cudaMemcpyHostToDevice);

    M_calculation <<< grid, block >>> (M_out_cuda, cuda_v1, cuda_v2, w, h, nc );
//cudaMemcpy(imgOut, M_out_cuda , w*h*nc*sizeof(float) , cudaMemcpyDeviceToHost);

    float *T_out_cuda;

    cudaMalloc((void**)&T_out_cuda , h*w*nc*sizeof(float));
    cudaMemcpy(T_out_cuda, imgOut , h*w*nc*sizeof(float) , cudaMemcpyHostToDevice);  

    cuda_convolution <<< grid, block >>> (M_out_cuda, T_out_cuda, kernel_cuda, w, h, nc, radius );

    cudaMemcpy(imgOut, T_out_cuda , w*h*nc*sizeof(float) , cudaMemcpyDeviceToHost);


    if (nc == 1){
        cv::Mat m11(h,w,CV_32FC1);
        convert_layered_to_mat(m11 , imgOut);
        showImage("m11", 10*m11, 100+w, 100);
    }
    else if (nc ==3){
        float *m11_flat = new float[(size_t)w*h*1];
        float *m12_flat = new float[(size_t)w*h*1];
        float *m22_flat = new float[(size_t)w*h*1];

        for (int j=0 ; j<h; j++){
            for (int i=0 ; i<w; i++){
                m11_flat[i + w*j] = imgOut[i + w*j + 0*w*h];
                m12_flat[i + w*j] = imgOut[i + w*j + 1*w*h];
                m22_flat[i + w*j] = imgOut[i + w*j + 2*w*h];  
            }
        }
        cv::Mat m11(h,w,CV_32FC1);
        cv::Mat m12(h,w,CV_32FC1);
        cv::Mat m22(h,w,CV_32FC1);
        convert_layered_to_mat(m11 , m11_flat);
        convert_layered_to_mat(m12 , m12_flat);
        convert_layered_to_mat(m22 , m22_flat);

        showImage("m11", 10*m11, 100+w, 100);
        showImage("m12", 10*m12, 100, 100+h);
        showImage("m22", 10*m22, 100+w, 100+h);  

        delete[] m11_flat;
        delete[] m12_flat; 
        delete[] m22_flat;
    }


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

    cudaFree(imgOut_cuda);
    cudaFree(imgIn_cuda);
    cudaFree(kernel_cuda);
    cudaFree(cuda_v1);
    cudaFree(cuda_v2);
    cudaFree(T_out_cuda);
    cudaFree(M_out_cuda);


    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



