	
typedef enum DWT{
  	FORWARD,
	REVERSE,
}DWT;

//Precompilation parameter that determine whether we compute the DWT 53 or the DWT 97. Equal to 1, DWT 53. If 0, DWT 97.
#if !defined(DWT53_or_DWT97)
	#define DWT53_or_DWT97				1
#endif

//Precompilation parameter that determines the number of decomposition levels of the DWT applied over the input image 
#if !defined(LEVELS)
	#define LEVELS		5
#endif

//Size of the number of columns that each thread computes (columns inside the Data block).
#define NELEMENTS_THREAD_X 			2

//Size of the lenght in samples of each Data block. The width is fixed to 32 threads per warp by 2 samples for each thread = 64.
#if !defined(NELEMENTS_THREAD_Y)
	#define NELEMENTS_THREAD_Y 	18
#endif

//Weights for the coefficients in each lifting step for the 5/3 and the 9/7, and normalization weights for the 9/7.
#define LIFTING_STEPS_I53_1 	0.5f 
#define LIFTING_STEPS_I53_2 	0.25f

#define LIFTING_STEPS_I97_1 	-1.586134342059924f 
#define LIFTING_STEPS_I97_2 	-0.052980118572961f
#define LIFTING_STEPS_I97_3  	0.882911075530934f
#define LIFTING_STEPS_I97_4 	0.443506852043971f

#define NORMALIZATION_I97_1 	1.230174104914001f
#define NORMALIZATION_I97_2		0.812893066f


//Thread block size used in the forward DWT
#if !defined(NTHREADSBLOCK_DWT_F)
	#define NTHREADSBLOCK_DWT_F		128
#endif

//Thread block size used in the reverse DWT
#if !defined(NTHREADSBLOCK_DWT_R)
	#define NTHREADSBLOCK_DWT_R		128
#endif

//Warp size fixed to 32 in the current CUDA architectures (Maxwell)
#define WARPSIZE	32

//Precompilation parameter that determines if shuffle instructions will be used to compute the horizontal filter. ONLY for Kepler or following architectures. If shuffle instructions are not used, an intermediate buffer in shared memory is employed instead.
#if !defined(SHUFFLE)
	#define SHUFFLE					1
#endif

//Precompilation parameter used to store the whole Data block in shared memory instead of using the registers to hold intermediate data.
#if !defined(FULLSHARED)
	#define FULLSHARED					0
#endif

#if FULLSHARED != 1
	#define SHARED_MEMORY_STRIDE 0
	#define VOLATILE		
#else
	#define SHARED_MEMORY_STRIDE ((NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X)+1)	
	#define VOLATILE volatile		
#endif

//The following precompilation parameters determine the sizes and number of the random generated images employed in the "main_example". Experiments start with an image of size EXPERIMENT_INI x EXPERIMENT_INI, and each of the followin experiments increase each dimension of this image by EXPERIMENT_INCREMENT.

#if !defined(NEXPERIMNETS)
	#define NEXPERIMNETS			10
#endif

#if !defined(EXPERIMENT_INI)
	#define EXPERIMENT_INI			1024
#endif

#if !defined(EXPERIMENT_INCREMENT)
	#define EXPERIMENT_INCREMENT	0
#endif

//Data types employed to represent each input sample. DATATYPE_16BITS_or_32BITS equal to 1 indicates that int16_t are used. If its 0, normal int/float of 32 bites are used. 
#if !defined(DATATYPE_16BITS_or_32BITS)
	#define DATATYPE_16BITS_or_32BITS 	1
#endif

//The following precompilation parameters can be used to deactivate the write, read, vertical filter or horizontal filter of the DWT, for profiling pourposes.
#if !defined(READ)
	#define READ 	1
#endif

#if !defined(WRITE)
	#define WRITE 	1
#endif

#if !defined(VERTICAL_COMPUTATION)
	#define VERTICAL_COMPUTATION 	1
#endif

#if !defined(HORIZONTAL_COMPUTATION)
	#define HORIZONTAL_COMPUTATION 	1
#endif

//The following precompilation parameters determine the data types employed in the device, depending on the use of the 5/3 or the 9/7 DWT. 

//If we compute the DWT 53
#if DWT53_or_DWT97 == 1
	#define REG_DATATYPE			int
	//In the DWT 5/3 each Data block is overlapped with its neighbours 4 samples 
	#define OVERLAP					4
	#if DATATYPE_16BITS_or_32BITS == 1
		#define DATATYPE			int16_t
		#define DATATYPE2 			int	
	#else
		#define DATATYPE 			int
		#define DATATYPE2 			int2
	#endif

//If we compute the DWT 97
#else
	#define REG_DATATYPE			float
//In the DWT 9/7 each Data block is overlapped with its neighbours 8 samples 
	#define OVERLAP					8
	#if DATATYPE_16BITS_or_32BITS == 1
		#define DATATYPE 			int16_t
		#define DATATYPE2 			int
	#else
		#define DATATYPE 			float
		#define DATATYPE2 			float2
	#endif
#endif

//Precompilation parameter used to assign an amount of shared memory to the kernels. Used for profiling pourposes to evaluate the impact of the occupancy in the application.
#if !defined(SYNTHETIC_SHARED)
	#define SYNTHETIC_SHARED 	0
#endif

