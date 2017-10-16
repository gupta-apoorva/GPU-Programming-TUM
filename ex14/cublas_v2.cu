#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
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


int main(int argc, char **argv)
{
	int n = 1000000;    
	float* a = new float[n];
	for (int i=0 ; i<n; i++){
	a[i] = i;
	}

	float* a_cuda;
	cudaMalloc((void**)&a_cuda, n*sizeof(float));
	cudaMemcpy(a_cuda, a , n*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cublasHandle_t handle;
	cublasCreate(&handle);

	float sum_l =0;
	cudaEventRecord(start);
	    cublasSasum(handle, n, a_cuda, 1, &sum_l);
	cudaEventRecord(stop);

	    
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "ARRAY SUM: " << sum_l  << " Time: "<< milliseconds << endl;


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cublasDestroy(handle);
	free(a);
	cudaFree(a_cuda);

	return 0;
}



