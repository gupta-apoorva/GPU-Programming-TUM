Exercise 1: Check CUDA and the installed GPU

1. Open a terminal and check whether CUDA is installed: nvcc --version. Which version
is installed?

	Ans: V8.0.44

2. Go to the “CUDA samples” folder 1 and run deviceQuery. Find out the following:
(a) name of the installed GPU and its compute capability (“CUDA Capability”)

	Ans: CC 6.1 , Nvidia Geforce GTX 1050Ti 

(b) number of multiprocessors and CUDA cores

	Ans: MP 6, cuda cores 768

(c) amount of global memory

	Ans: 4 GB

(d) max. amount of registers and shared memory per block

	Ans: registers: 65536 Shared Memory per block: 49152 bytes\

	Device 0: "GeForce GTX 1050 Ti"
	  CUDA Driver Version / Runtime Version          8.0 / 7.0
	  CUDA Capability Major/Minor version number:    6.1
	  Total amount of global memory:                 4038 MBytes (4234608640 bytes)
	MapSMtoCores for SM 6.1 is undefined.  Default to use 128 Cores/SM
	MapSMtoCores for SM 6.1 is undefined.  Default to use 128 Cores/SM
	  ( 6) Multiprocessors, (128) CUDA Cores/MP:     768 CUDA Cores
	  GPU Max Clock rate:                            1418 MHz (1.42 GHz)
	  Memory Clock rate:                             3504 Mhz
	  Memory Bus Width:                              128-bit
	  L2 Cache Size:                                 1048576 bytes
	  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
	  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
	  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
	  Total amount of constant memory:               65536 bytes
	  Total amount of shared memory per block:       49152 bytes
	  Total number of registers available per block: 65536
	  Warp size:                                     32
	  Maximum number of threads per multiprocessor:  2048
	  Maximum number of threads per block:           1024
	  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
	  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
	  Maximum memory pitch:                          2147483647 bytes
	  Texture alignment:                             512 bytes
	  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
	  Run time limit on kernels:                     Yes
	  Integrated GPU sharing Host Memory:            No
	  Support host page-locked memory mapping:       Yes
	  Alignment requirement for Surfaces:            Yes
	  Device has ECC support:                        Disabled
	  Device supports Unified Addressing (UVA):      Yes
	  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0

