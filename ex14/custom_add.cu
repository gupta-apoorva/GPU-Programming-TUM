#include <cuda_runtime.h>
#include <iostream>
#include "cublas.h"
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


__global__ void summing(float* a_cuda, float* b_cuda, int n){
    int t_numx = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float blocksum;
if (tid ==0){
blocksum = 0;}

__syncthreads();

if (t_numx <n){
    atomicAdd(&blocksum, a_cuda[t_numx]);
}
__syncthreads();
if (tid ==0){    
    b_cuda[blockIdx.x] += blocksum;
}
}


int main(int argc, char **argv)
{
	int n = 1000000;    
	float* a = new float[n];
	for (int i=0 ; i<n; i++){
		a[i] = i;
	}

	float* a_cuda, *b_cuda;
	cudaMalloc((void**)&a_cuda, n*sizeof(float));
	cudaMemcpy(a_cuda, a , n*sizeof(float), cudaMemcpyHostToDevice);

	int block = 1024;
	int grid = ((block + n -1)/block);

	cudaMalloc((void**)&b_cuda,  grid*sizeof(float));
	cudaMemset(b_cuda, 0 , grid*sizeof(float));   

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* b = new float[grid]; 
	float sum_l =0;
	cudaEventRecord(start);

	summing <<<grid, block>>> (a_cuda, b_cuda , n);

	cudaMemcpy(b , b_cuda, grid*sizeof(float) , cudaMemcpyDeviceToHost);
	for (int i=0 ; i<grid ; i++){
		sum_l +=b[i];
	}

	cudaEventRecord(stop);

    
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "ARRAY SUM: " << sum_l  << " Time: "<< milliseconds << endl;
	

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(a);
	free(b);
	cudaFree(a_cuda);
	cudaFree(b_cuda);

return 0;
}



