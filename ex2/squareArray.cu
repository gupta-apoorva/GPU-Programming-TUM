// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include <cuda_runtime.h>
#include <iostream>
using namespace std;


// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__global__ void square(float* cuda_a, int n){
    int t_numx = threadIdx.x + blockIdx.x*blockDim.x;
    int t_numy = threadIdx.y + blockIdx.y*blockDim.y;
    int t_numz = threadIdx.z + blockIdx.z*blockDim.z;
    if (t_numx + t_numy + t_numz < n){
        cuda_a[t_numx + t_numy + t_numz] = cuda_a[t_numx + t_numy + t_numz]*cuda_a[t_numx + t_numy + t_numz];
    }
}


int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *a = new float[n];
    for(int i=0; i<n; i++) a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = a[i];
        val = val*val;
        a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;
    


    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) a[i] = i;


    float *cuda_a;
    cudaMalloc((void**)&cuda_a, n*sizeof(float));
    cudaMemcpy(cuda_a, a, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block = dim3(128,1,1);
    int grid_x = ((n + block.x + 1)/block.x);
    int grid_y = 1;
    int grid_z = 1;
    dim3 grid = dim3(grid_x, grid_y, grid_z );

    square <<<block,grid>>> (cuda_a, n);

    cudaMemcpy(a, cuda_a, n*sizeof(int), cudaMemcpyDeviceToHost);

    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << a[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] a;
    cudaFree(cuda_a);
}



