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

int index = t_numx + w*t_numy + w*h*t_numz;
    if (t_numx + 1 < w && t_numy < h && t_numz < nc){
        cuda_v1[index] = cuda_imgIn[index +1] - cuda_imgIn[index];
    }

    if (t_numx < w && t_numy + 1< h && t_numz < nc){
        cuda_v2[index] = cuda_imgIn[index + w] - cuda_imgIn[index];
    }
}

__global__ void divergence(float* cuda_div , float* cuda_v1, float* cuda_v2, int w, int h, int nc ){
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
}

__global__ void l2_norm(float* cuda_l2, float* cuda_div, int w, int h, int nc ){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;

	int index = t_numx + w*t_numy + w*h*t_numz;
	float norm = 0;
    if (t_numx < w && t_numy < h && t_numz == 0){
        for (int i=0 ; i< nc; i++){
            norm += pow(cuda_div[index], 2);
        }
        cuda_l2[index] = sqrtf(norm);
    }
}


int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;

    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
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
    
    //cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
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


//GPU version

    int array_size = w*h*nc;
    
    float* cuda_imgIn;
    float* cuda_v1;
    float* cuda_v2;


    cudaMalloc((void**) &cuda_imgIn , array_size*sizeof(float));
    cudaMalloc((void**) &cuda_v1, array_size*sizeof(float));
    cudaMalloc((void**) &cuda_v2, array_size*sizeof(float));

    cudaMemcpy(cuda_imgIn, imgIn , array_size*sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemset(cuda_v1, 0 , array_size*sizeof(float));
    cudaMemset(cuda_v2, 0 , array_size*sizeof(float));

    dim3 block = dim3(32,32,1);
    int grid_x = ((w + block.x - 1)/block.x);
    int grid_y = ((h + block.y - 1)/block.y);
    int grid_z = ((nc + block.z - 1)/block.z);
    dim3 grid = dim3(grid_x, grid_y, grid_z );

    gradient <<<grid, block>>>(cuda_imgIn, cuda_v1, cuda_v2, w, h, nc );

    float *cuda_div;
    cudaMalloc((void**) &cuda_div , array_size*sizeof(float));
    cudaMemset(cuda_div, 0 , array_size*sizeof(float));

    divergence <<< grid , block>>> (cuda_div , cuda_v1, cuda_v2, w, h, nc );

    float *cuda_l2;

    cudaMalloc((void**) &cuda_l2 , w*h*sizeof(float));
    int laplacian_size = w*h;
    float* laplacian = new float[laplacian_size];

    cudaMemset(cuda_l2, 0 , laplacian_size*sizeof(float));

    l2_norm <<<grid, block>>> (cuda_l2, cuda_div, w, h, nc);

    cudaMemcpy(laplacian, cuda_l2 , laplacian_size*sizeof(float) , cudaMemcpyDeviceToHost);

// GPU version end

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, laplacian);
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
    delete[] laplacian;
    cudaFree(cuda_imgIn);
    cudaFree(cuda_div);
    cudaFree(cuda_l2);
    cudaFree(cuda_v1);
    cudaFree(cuda_v2);


    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



