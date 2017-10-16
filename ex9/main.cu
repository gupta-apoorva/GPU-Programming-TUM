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

__global__ void gradient(float* cuda_imgIn,float* cuda_v1,float* cuda_v2,int w,int h,int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

    if (t_numx + 1 < w && t_numy < h && t_numz < nc){
        cuda_v1[t_numx + w*t_numy + w*h*t_numz] = cuda_imgIn[t_numx + 1 + w*t_numy + w*h*t_numz] - cuda_imgIn[t_numx + w*t_numy + w*h*t_numz];
    }

    if (t_numx < w && t_numy + 1< h && t_numz < nc){
        cuda_v2[t_numx + w*t_numy + w*h*t_numz] = cuda_imgIn[t_numx + w*(t_numy+1) + w*h*t_numz] - cuda_imgIn[t_numx + w*t_numy + w*h*t_numz];
    }
}

__global__ void calculate_g_and_multiply_simple(float* cuda_v1, float* cuda_v2, int w, int h, int nc){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = 0;

    float g = 1;
    if (t_numx < w && t_numy < h && t_numz < nc){
        for (int k=0 ; k<nc ; k++){
            cuda_v1[t_numx + w*t_numy + w*h*k] = g*cuda_v1[t_numx + w*t_numy + w*h*k];
            cuda_v2[t_numx + w*t_numy + w*h*k] = g*cuda_v2[t_numx + w*t_numy + w*h*k];
        }
    }
}


__global__ void calculate_g_and_multiply_max(float* cuda_v1, float* cuda_v2, int w, int h, int nc, float epsilon){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = 0;

    float s=0;
    float g = 0;
    if (t_numx < w && t_numy < h && t_numz < nc){
        for (int k=0 ; k<nc ; k++){
            s += pow(cuda_v1[t_numx + w*t_numy + w*h*k] , 2) + pow(cuda_v2[t_numx + w*t_numy + w*h*k],2); 
        }
        s = sqrt(s);
        g = 1.f/max(epsilon, s);
        for (int k=0 ; k<nc ; k++){
            cuda_v1[t_numx + w*t_numy + w*h*k] = g*cuda_v1[t_numx + w*t_numy + w*h*k];
            cuda_v2[t_numx + w*t_numy + w*h*k] = g*cuda_v2[t_numx + w*t_numy + w*h*k];
        }
    }
}

__global__ void calculate_g_and_multiply_exp(float* cuda_v1, float* cuda_v2, int w, int h, int nc, float epsilon){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = 0;

    float s=0;
    float g = 0;
    if (t_numx < w && t_numy < h && t_numz < nc){
        for (int k=0 ; k<nc ; k++){
            s += pow(cuda_v1[t_numx + w*t_numy + w*h*k] , 2) + pow(cuda_v2[t_numx + w*t_numy + w*h*k],2); 
        }
        s = sqrt(s);
        g = exp(-pow(s,2)/epsilon)/epsilon;
        for (int k=0 ; k<nc ; k++){
            cuda_v1[t_numx + w*t_numy + w*h*k] = g*cuda_v1[t_numx + w*t_numy + w*h*k];
            cuda_v2[t_numx + w*t_numy + w*h*k] = g*cuda_v2[t_numx + w*t_numy + w*h*k];
        }
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

    if (t_numy > 0 && t_numx < w && t_numy < h && t_numz < nc){
        cuda_div[index] += cuda_v2[index] - cuda_v2[index - w];
    }

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
    int iterations = 0;
    float tau = 0.0;
    float epsilon = 0.0;
    int g_type = 0;
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    bool ret2 = getParam("iter", iterations, argc, argv);
    if (!ret2) {cerr << "ERROR: Num of iterations not specified" << endl; return 1;}
    bool ret3 = getParam("tau", tau, argc, argv);
    if (!ret3) {cerr << "ERROR: no tau specified" << endl; return 1;}
    bool ret4 = getParam("epsilon", epsilon, argc, argv);
    if (!ret4) {cerr << "ERROR: no epsilon specified" << endl; return 1;}
    bool ret5 = getParam("g_type", g_type, argc, argv);
    if (!ret5) {cerr << "ERROR: no gradient calculation type specified" << endl; return 1;}
    if (argc <= 4) { cout << "Usage: " << argv[0] << " -i <image> -iter <iterations> -tau <tau> -epsilon <epsilon> -g_type <g_type>[-repeats <repeats>] [-gray]" << endl; return 1; }
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
    
    cv::Mat mOut(h,w,mIn.type());

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA

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


//GPU version

    int array_size = w*h*nc;
    //int iterations = 100;
    //float tau = 0.2;
    //float epsilon = 0.1;
    
    float* cuda_imgIn;
    float* cuda_v1;
    float* cuda_v2;


    cudaMalloc((void**) &cuda_imgIn , array_size*sizeof(float));
    cudaMalloc((void**) &cuda_v1, array_size*sizeof(float));
    cudaMalloc((void**) &cuda_v2, array_size*sizeof(float));

    cudaMemcpy(cuda_imgIn, imgIn , array_size*sizeof(float) , cudaMemcpyHostToDevice);
    

    float *cuda_div;
    cudaMalloc((void**) &cuda_div , array_size*sizeof(float));


    dim3 block = dim3(32,32,1);
    int grid_x = ((w + block.x - 1)/block.x);
    int grid_y = ((h + block.y - 1)/block.y);
    int grid_z = ((nc + block.z - 1)/block.z);
    dim3 grid = dim3(grid_x, grid_y, grid_z );


    for (int iter=0; iter<iterations ; iter++){

    cudaMemset(cuda_v1, 0 , array_size*sizeof(float));
    cudaMemset(cuda_v2, 0 , array_size*sizeof(float));
    gradient <<<grid, block>>>(cuda_imgIn, cuda_v1, cuda_v2, w, h, nc );

    if (g_type == 0)
        calculate_g_and_multiply_simple <<< grid, block>>>(cuda_v1, cuda_v2, w, h, nc);
    else if (g_type == 1)
        calculate_g_and_multiply_max <<< grid, block>>>(cuda_v1, cuda_v2, w, h, nc, epsilon);
    else if (g_type ==2)
        calculate_g_and_multiply_exp <<< grid, block>>>(cuda_v1, cuda_v2, w, h, nc, epsilon);

    cudaMemset(cuda_div, 0 , array_size*sizeof(float));
    divergence_update <<<grid, block>>> (cuda_imgIn, cuda_div , cuda_v1, cuda_v2, w, h, nc, tau );
    
    }
    cudaMemcpy(imgOut, cuda_imgIn , array_size*sizeof(float) , cudaMemcpyDeviceToHost);

// GPU version end


    // show input image
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

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    cudaFree(cuda_v1);
    cudaFree(cuda_v2);
    cudaFree(cuda_div);
    cudaFree(cuda_imgIn);

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



