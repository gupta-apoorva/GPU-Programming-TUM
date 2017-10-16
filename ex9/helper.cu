// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###
// ###
// ###
// ### THIS FILE IS SUPPOSED TO REMAIN UNCHANGED
// ###
// ###


#include "helper.h"
#include <cstdlib>
#include <iostream>
using std::stringstream;
using std::cerr;
using std::cout;
using std::endl;
using std::string;




// parameter processing: template specialization for T=bool
template<>
bool getParam<bool>(std::string param, bool &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc) || argv[i+1][0]=='-') { var = true; return true; }
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}




// opencv helpers
void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
            }
        }
    }
}
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn)
{
    convert_layered_to_interleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
}


void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc)
{
    if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
    size_t nOmega = (size_t)w*h;
    for (int y=0; y<h; y++)
    {
        for (int x=0; x<w; x++)
        {
            for (int c=0; c<nc; c++)
            {
                aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
            }
        }
    }
}
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn)
{
    convert_interleaved_to_layered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
}



void showImage(string title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}

void showHistogram256(const char *windowTitle, int *histogram, int windowX, int windowY)
{
  const int nbins = 256;
  cv::Mat canvas = cv::Mat::ones(125, 512, CV_8UC3);

  float hmax = 0;
  for(int i = 0; i < nbins; ++i)
    hmax = max((int)hmax, histogram[i]);

  for (int j = 0, rows = canvas.rows; j < nbins-1; j++)
  {
    for(int i = 0; i < 2; ++i)
      cv::line(
        canvas, 
        cv::Point(j*2+i, rows), 
        cv::Point(j*2+i, rows - (histogram[j] * 125.0f) / hmax), 
        cv::Scalar(255,128,0), 
        1, 8, 0
        );
  }

  showImage(windowTitle, canvas, windowX, windowY);
}




// adding Gaussian noise
float noise(float sigma)
{
    float x1 = (float)rand()/RAND_MAX;
    float x2 = (float)rand()/RAND_MAX;
    return sigma * sqrtf(-2*log(std::max(x1,0.000001f)))*cosf(2*M_PI*x2);
}
void addNoise(cv::Mat &m, float sigma)
{
    float *data = (float*)m.data;
    int w = m.cols;
    int h = m.rows;
    int nc = m.channels();
    size_t n = (size_t)w*h*nc;
    for(size_t i=0; i<n; i++)
    {
        data[i] += noise(sigma);
    }
}




// cuda error checking
string prev_file = "";
int prev_line = 0;
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
        exit(1);
    }
    prev_file = file;
    prev_line = line;
}
