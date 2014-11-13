#include <stdio.h>
#include "RBased_DWT_common.h"



/*


Functions in this code are organized in 7 sections. 2 Sections corresponds to functions executed in the host and the other 5 are functions used in the device. 

Host functions sections:

1. PRE/POST COMPUTE FUNCTIONS: Allocate and release device memory.
2. DWT FUNCTIONS: Precomputing needed to apply N levels of the DWT over an input, launch a kernel for each DWT level.

Device functions sections:

3. CUDA KERNELS: Each CUDA kernel computes a single DWT level (vertical + horizontal filter).
4. PRE-COMPUTE FUNCTIONS: Inline device functions used by the kernels to (mainly) compute in which input coordinates each warp has to work (asign a data block to a warp).
5. DATA MANAGEMENT FUNCTIONS: Inline device functions used by the kernels to read (or write) a data block from the device main memory to the registers, and the other way around.
6. FILTER COMPUTATION FUNCTIONS: Inline device functions used by the kernels to compute the vertical or horizontal filter over a full data block.
7. FILTER KERNEL FUNCTIONS:  Inline device functions used by the kernels to compute a lifting step operation over 3 samples.		


Example of function call flow:

	<HOST> PRE/POST COMPUTE FUNCTIONS 
	|
	|__ <HOST> DWT FUNCTIONS 
		|
		|__ <DEVICE> CUDA KERNELS 	
			|
			|__ <DEVICE> PRE-COMPUTE FUNCTIONS 	
				<DEVICE> DATA MANAGEMENT FUNCTIONS 		
				<DEVICE> FILTER COMPUTATION FUNCTIONS 	
				|
				|__ <DEVICE> FILTER KERNEL FUNCTIONS 				

*/




/**************************************************************

START - <DEVICE> FILTER KERNEL FUNCTIONS

**************************************************************/

//CDF 5/3 (1st Lifting Step) - FORWARD 
inline __device__ void LStep_1_53_F(int a, VOLATILE int* b, int c){ *b -= ((a+c)>>1);}

//CDF 5/3 (2nd Lifting Step) - FORWARD 
inline __device__ void LStep_2_53_F(int a, VOLATILE int* b, int c){ *b += ((a + c + 2)>>2);}
		

//CDF 5/3 (1st Lifting Step) - REVERSE
inline __device__ void LStep_1_53_R(int a, VOLATILE int* b, int c){ *b -= (((a + c + 2))>>2);}

//CDF 5/3 (2nd Lifting Step) - REVERSE
inline __device__ void LStep_2_53_R(int a, VOLATILE int* b, int c){ *b += (((a+c)>>1));}



//CDF 9/7 (1st Lifting Step) - FORWARD 
inline __device__ void LStep_1_97_F(float a, VOLATILE float* b, float c){ *b += ((a + c)* LIFTING_STEPS_I97_1);}

//CDF 9/7 (2nd Lifting Step) - FORWARD 
inline __device__ void LStep_2_97_F(float a, VOLATILE float* b, float c){ *b += ((a + c)* LIFTING_STEPS_I97_2);}

//CDF 9/7 (3rd Lifting Step) - FORWARD 
inline __device__ void LStep_3_97_F(float a, VOLATILE float* b, float c){ *b += ((a + c)* LIFTING_STEPS_I97_3);}

//CDF 9/7 (4th Lifting Step) - FORWARD + normalization
inline __device__ void LStep_4_97_F(float a, VOLATILE float* b, float c){ *b = (*b + ((a + c)* LIFTING_STEPS_I97_4))*NORMALIZATION_I97_2;}	
		


//CDF 9/7 (1st Lifting Step) - REVERSE + normalization
inline __device__ void LStep_1_97_R(float a, VOLATILE float* b, float c){ *b = (*b/NORMALIZATION_I97_2) - ((a + c)* LIFTING_STEPS_I97_4);}

//CDF 9/7 (2nd Lifting Step) - REVERSE
inline __device__ void LStep_2_97_R(float a, VOLATILE float* b, float c){ *b -= ((a + c)* LIFTING_STEPS_I97_3);}

//CDF 9/7 (3rd Lifting Step) - REVERSE
inline __device__ void LStep_3_97_R(float a, VOLATILE float* b, float c){ *b -= ((a + c)* LIFTING_STEPS_I97_2);}

//CDF 9/7 (4th Lifting Step) - REVERSE
inline __device__ void LStep_4_97_R(float a, VOLATILE float* b, float c){ *b -= ((a + c)* LIFTING_STEPS_I97_1);}

//END - <DEVICE> FILTER KERNEL FUNCTIONS -----------------------------------------------------------------------













/**************************************************************
//START - <DEVICE> FILTER COMPUTATION FUNCTIONS
**************************************************************/



//VERTICAL FILTER FUNCTIONS , generic for all versions <shuffle instructions, shared memory with auxiliary buffer or full shared memory>	




inline __device__ void Vertical_Filter_Forward_53(int* TData,int TDSize_Y, int TDSize_X)
{
	int TDSize_Y_index	= 0;

	for(int TDSize_X_index = 0; TDSize_X_index < TDSize_X; TDSize_X_index++)
	{	
		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{			
			LStep_1_53_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}			
		
		LStep_1_53_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)]);		
		TDSize_Y_index = 0;
		LStep_2_53_F(TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)]);
		
		for(TDSize_Y_index = 2; TDSize_Y_index < TDSize_Y; TDSize_Y_index += 2)
		{
			LStep_2_53_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)]);
		}			
	}
}

inline __device__ void Vertical_Filter_Reverse_53(int* TData,int TDSize_Y, int TDSize_X)
{
	int TDSize_Y_index	= 0;
	
	for(int TDSize_X_index = 0; TDSize_X_index < TDSize_X; TDSize_X_index++)
	{			
		TDSize_Y_index	= 0;
		LStep_1_53_R(TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		
		for(TDSize_Y_index = 2; TDSize_Y_index < TDSize_Y; TDSize_Y_index += 2)
		{
			LStep_1_53_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{			
			LStep_2_53_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}				
		
		LStep_2_53_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);			
	}
}

inline __device__ void Vertical_Filter_Forward_97(float* TData, int TDSize_Y, int TDSize_X)
{

	int TDSize_Y_index	= 0;

	for(int TDSize_X_index = 0; TDSize_X_index < TDSize_X; TDSize_X_index++)
	{	
		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{
			LStep_1_97_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}
						
		LStep_1_97_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		TDSize_Y_index = 0;
		LStep_2_97_F(TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		
		for(TDSize_Y_index = 2; TDSize_Y_index < TDSize_Y; TDSize_Y_index += 2)
		{
			LStep_2_97_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{
			LStep_3_97_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}			
		
		LStep_3_97_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);		
		TDSize_Y_index = 0;
		LStep_4_97_F(TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);

		
		for(TDSize_Y_index = 2; TDSize_Y_index < TDSize_Y; TDSize_Y_index += 2)
		{
			LStep_4_97_F(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{
			TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))] *= NORMALIZATION_I97_1;
		}

		TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))] *= NORMALIZATION_I97_1;			
	}
}

inline __device__ void Vertical_Filter_Reverse_97(float* TData, int TDSize_Y, int TDSize_X)
{

	int TDSize_Y_index	= 0;

	for(int TDSize_X_index = 0; TDSize_X_index < TDSize_X; TDSize_X_index++)
	{
		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{
			TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))] /= NORMALIZATION_I97_1;
		}

		TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))] /= NORMALIZATION_I97_1;
		TDSize_Y_index	= 0;
		LStep_1_97_R(TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		
		for(TDSize_Y_index = 2; TDSize_Y_index < TDSize_Y; TDSize_Y_index += 2)
		{
			LStep_1_97_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}

		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{			
			LStep_2_97_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}				
		
		LStep_2_97_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);		
		TDSize_Y_index	= 0;
		LStep_3_97_R(TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		
		for(TDSize_Y_index = 2; TDSize_Y_index < TDSize_Y; TDSize_Y_index += 2)
		{
			LStep_3_97_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}	
		
		for(TDSize_Y_index = 1; TDSize_Y_index < (TDSize_Y-1); TDSize_Y_index += 2)
		{			
			LStep_4_97_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index+1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);
		}				
		
		LStep_4_97_R(TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))], TData[((TDSize_Y_index-1)*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*(threadIdx.x))]);					
	}
}




//HORIZONTAL FILTER FUNCTIONS , specific for each version <shuffle instructions, shared memory with auxiliary buffer or full shared memory>		


//SHARED MEMORY W/ AUXILIARY BUFFER - HORIZONTAL FILTER FUNCTIONS


inline __device__ void Horizontal_Filter_Forward_53_Shared(int* TData, int* Shared_Data, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){	
		Shared_Data[threadIdx.x] = TData[TDSize_Y_index*2];
		LStep_1_53_F(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], Shared_Data[(((threadIdx.x)%32)==31?threadIdx.x:threadIdx.x+1)]);
	}

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){		
		Shared_Data[threadIdx.x] = TData[(TDSize_Y_index*2)+1];
		LStep_2_53_F(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], Shared_Data[(((threadIdx.x)%32)==0?threadIdx.x:threadIdx.x-1)]);
	}
}

inline __device__ void Horizontal_Filter_Reverse_53_Shared(int* TData, int* Shared_Data, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){	
		Shared_Data[threadIdx.x] = TData[(TDSize_Y_index*2)+1];
		LStep_1_53_R(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], Shared_Data[(((threadIdx.x)%32)==0?threadIdx.x:threadIdx.x-1)]);
	}

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){
		Shared_Data[threadIdx.x] = TData[TDSize_Y_index*2];
		LStep_2_53_R(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], Shared_Data[(((threadIdx.x)%32)==31?threadIdx.x:threadIdx.x+1)]);
	}
}

inline __device__ void Horizontal_Filter_Forward_97_Shared(float* TData, float* Shared_Data, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){

		Shared_Data[threadIdx.x] = TData[TDSize_Y_index*2];
		LStep_1_97_F(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], Shared_Data[(((threadIdx.x)%32)==31?threadIdx.x:threadIdx.x+1)]);
	
		Shared_Data[threadIdx.x] = TData[(TDSize_Y_index*2)+1];
		LStep_2_97_F(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], Shared_Data[(((threadIdx.x)%32)==0?threadIdx.x:threadIdx.x-1)]);

		Shared_Data[threadIdx.x] = TData[TDSize_Y_index*2];	
		LStep_3_97_F(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], Shared_Data[(((threadIdx.x)%32)==31?threadIdx.x:threadIdx.x+1)]);

		Shared_Data[threadIdx.x] = TData[(TDSize_Y_index*2)+1];
		LStep_4_97_F(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], Shared_Data[(((threadIdx.x)%32)==0?threadIdx.x:threadIdx.x-1)]);

		TData[(TDSize_Y_index*2)+TDSize_X_index] *= NORMALIZATION_I97_1;
	}		
}

inline __device__ void Horizontal_Filter_Reverse_97_Shared(float* TData, float* Shared_Data, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){
	
		TData[(TDSize_Y_index*2)+TDSize_X_index] /= NORMALIZATION_I97_1;

		Shared_Data[threadIdx.x] = TData[(TDSize_Y_index*2)+1];	
		LStep_1_97_R(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], Shared_Data[(((threadIdx.x)%32)==0?threadIdx.x:threadIdx.x-1)]);

		Shared_Data[threadIdx.x] = TData[TDSize_Y_index*2];
		LStep_2_97_R(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], Shared_Data[(((threadIdx.x)%32)==31?threadIdx.x:threadIdx.x+1)]);

		Shared_Data[threadIdx.x] = TData[(TDSize_Y_index*2)+1];
		LStep_3_97_R(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], Shared_Data[(((threadIdx.x)%32)==0?threadIdx.x:threadIdx.x-1)]);

		Shared_Data[threadIdx.x] = TData[TDSize_Y_index*2];
		LStep_4_97_R(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], Shared_Data[(((threadIdx.x)%32)==31?threadIdx.x:threadIdx.x+1)]);	
	}	
}



//FULL SHARED MEMORY - HORIZONTAL FILTER FUNCTIONS




inline __device__ void Horizontal_Filter_Forward_53_Full_Shared(VOLATILE int* TData, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)
		LStep_1_53_F(TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], &(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)]), TData[(TDSize_Y_index*2)+(((threadIdx.x)%32)==31?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x+1)))]);

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)	
		LStep_2_53_F(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+TDSize_X_index+(((threadIdx.x)%32)==0?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x-1)))]);

}

inline __device__ void Horizontal_Filter_Reverse_53_Full_Shared(VOLATILE int* TData, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)	
		LStep_1_53_R(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+TDSize_X_index+(((threadIdx.x)%32)==0?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x-1)))]);

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)
		LStep_2_53_R(TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+(((threadIdx.x)%32)==31?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x+1)))]);
}

inline __device__ void Horizontal_Filter_Forward_97_Full_Shared(VOLATILE float* TData, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){

		LStep_1_97_F(TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], &(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)]), TData[(TDSize_Y_index*2)+(((threadIdx.x)%32)==31?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x+1)))]);
		LStep_2_97_F(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+TDSize_X_index+(((threadIdx.x)%32)==0?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x-1)))]);
		LStep_3_97_F(TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], &(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)]), TData[(TDSize_Y_index*2)+(((threadIdx.x)%32)==31?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x+1)))]);
		LStep_4_97_F(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+TDSize_X_index+(((threadIdx.x)%32)==0?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x-1)))]);
	
		TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)] *= NORMALIZATION_I97_1;
	}	
}

inline __device__ void Horizontal_Filter_Reverse_97_Full_Shared(VOLATILE float* TData, int TDSize_Y, int TDSize_X)
{
	int TDSize_X_index = TDSize_X>>1;

	for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){
	
		TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)] /= NORMALIZATION_I97_1;

		LStep_1_97_R(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+TDSize_X_index+(((threadIdx.x)%32)==0?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x-1)))]);
		LStep_2_97_R(TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+(((threadIdx.x)%32)==31?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x+1)))]);
		LStep_3_97_R(TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+TDSize_X_index+(((threadIdx.x)%32)==0?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x-1)))]);
		LStep_4_97_R(TData[(TDSize_Y_index*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], &TData[(TDSize_Y_index*2)+TDSize_X_index+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[(TDSize_Y_index*2)+(((threadIdx.x)%32)==31?(SHARED_MEMORY_STRIDE*threadIdx.x):(SHARED_MEMORY_STRIDE*(threadIdx.x+1)))]);	
	}		
}




//SHUFFLE INSTRUCTIONS - HORIZONTAL FILTER FUNCTIONS




#if SHUFFLE == 1	

	inline __device__ void Horizontal_Filter_Forward_53_Shuffle(int* TData, int TDSize_Y, int TDSize_X)
	{
		int TDSize_X_index = TDSize_X>>1;

		for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)
			LStep_1_53_F(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], __shfl_down(TData[TDSize_Y_index*2], 1));

		for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)
			LStep_2_53_F(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], __shfl_up(TData[(TDSize_Y_index*2)+TDSize_X_index], 1));
	}

	inline __device__ void Horizontal_Filter_Reverse_53_Shuffle(int* TData, int TDSize_Y, int TDSize_X)
	{
		int TDSize_X_index = TDSize_X>>1;

		for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)		
			LStep_1_53_R(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], __shfl_up(TData[(TDSize_Y_index*2)+TDSize_X_index], 1));

		for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++)
			LStep_2_53_R(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], __shfl_down(TData[TDSize_Y_index*2], 1));	
	}

	inline __device__ void Horizontal_Filter_Forward_97_Shuffle(float* TData, int TDSize_Y, int TDSize_X)
	{
		int TDSize_X_index = TDSize_X>>1;

		for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){
			LStep_1_97_F(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], __shfl_down(TData[TDSize_Y_index*2], 1));
			LStep_2_97_F(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], __shfl_up(TData[(TDSize_Y_index*2)+TDSize_X_index], 1));
			LStep_3_97_F(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], __shfl_down(TData[TDSize_Y_index*2], 1));
			LStep_4_97_F(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], __shfl_up(TData[(TDSize_Y_index*2)+TDSize_X_index], 1));
	
			TData[(TDSize_Y_index*2)+TDSize_X_index] *= NORMALIZATION_I97_1;
		}		
	}

	inline __device__ void Horizontal_Filter_Reverse_97_Shuffle(float* TData, int TDSize_Y, int TDSize_X)
	{
		int TDSize_X_index = TDSize_X>>1;

		for(int TDSize_Y_index = 0; TDSize_Y_index < TDSize_Y; TDSize_Y_index ++){
		
			TData[(TDSize_Y_index*2)+TDSize_X_index] /= NORMALIZATION_I97_1;

			LStep_1_97_R(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], __shfl_up(TData[(TDSize_Y_index*2)+TDSize_X_index], 1));
			LStep_2_97_R(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], __shfl_down(TData[TDSize_Y_index*2], 1));
			LStep_3_97_R(TData[(TDSize_Y_index*2)+TDSize_X_index], &TData[TDSize_Y_index*2], __shfl_up(TData[(TDSize_Y_index*2)+TDSize_X_index], 1));
			LStep_4_97_R(TData[TDSize_Y_index*2], &TData[(TDSize_Y_index*2)+TDSize_X_index], __shfl_down(TData[TDSize_Y_index*2], 1));	
		}		
	}

#endif

//END - <DEVICE> FILTER COMPUTATION FUNCTIONS -----------------------------------------------------------------------














//START - <DEVICE> DATA MANAGEMENT FUNCTIONS -----------------------------------------------------------------------

inline __device__ void UpdateSubbandsCoordinates(int DSize_Current_X, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH)
{
	*TCoordinate_LL += DSize_Current_X;
	*TCoordinate_HL += DSize_Current_X;
	*TCoordinate_LH += DSize_Current_X;
}

inline __device__ void UpdateSubbandsCoordinates_LLaux(int DSize_Current_X, int DSize_Initial_X, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH)
{
	*TCoordinate_LL += (DSize_Current_X>>1);
	*TCoordinate_HL += DSize_Initial_X;
	*TCoordinate_LH += DSize_Initial_X;
}

inline __device__ void UpdateSubbandsCoordinates_Scheduler(int DSize_Current_X, int DSize_Initial_X, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int Last_Level)
{
	if(Last_Level)	UpdateSubbandsCoordinates(DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH);
	else			UpdateSubbandsCoordinates_LLaux(DSize_Current_X, DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH);
}

inline __device__ void ReadBlock(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate)
{

	for(int y = 0; y < TDSize_Y; y++){

		TData[(y<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data[*TCoordinate];
		TData[(y<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data[(*TCoordinate)+1];

		*TCoordinate += DSize_Current_X;
	}
}

inline __device__ void ReadBlock2(DATATYPE2* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate)
{

	*(TCoordinate)>>=1;
	for(int y = 0; y < TDSize_Y; y++){
		
		#if DATATYPE_16BITS_or_32BITS == 1

			TData[(y*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = 		Data[*TCoordinate];
			TData[(y*2)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = 	((int)TData[(y*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)])>>16;	
			TData[(y*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = 		(DATATYPE)TData[(y*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)];		
		
		#else

			TData[(y*2)] = 		Data[*TCoordinate].x;
			TData[(y*2)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = 	Data[*TCoordinate].y;
			
		#endif

		*TCoordinate += DSize_Current_X;
	}
	*(TCoordinate)<<=1;
}

inline __device__ void WriteSubbands(DATATYPE* Data, int DSize_Initial_X, int DSize_Current_X, REG_DATATYPE* TData, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int* index, int Last_Level){

	Data[*TCoordinate_LL] = TData[((*index)<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)];
	Data[*TCoordinate_HL] = TData[((*index)<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)];

	++(*index);

	Data[*TCoordinate_LH] = TData[((*index)<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)];
	Data[*TCoordinate_LH + (DSize_Current_X>>1)] = TData[((*index)<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)];

	UpdateSubbandsCoordinates_Scheduler(DSize_Current_X, DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, Last_Level);
}

inline __device__ void WriteSubbands_Top(DATATYPE* Data, int DSize_Current_X,  int DSize_Initial_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int Last_Level, int Overlap)
{	
	for(int y = 0; y < (TDSize_Y - (Overlap>>1)); y++)	
		WriteSubbands(Data, DSize_Initial_X, DSize_Current_X, TData, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, &y, Last_Level);			
}

inline __device__ void WriteSubbands_Middle(DATATYPE* Data, int DSize_Current_X,  int DSize_Initial_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int Last_Level, int Overlap)
{	
	for(int y = 0; y < (Overlap>>1); y+=2)
		UpdateSubbandsCoordinates_Scheduler(DSize_Current_X, DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, Last_Level);
	
	for(int y = (Overlap>>1); y < (TDSize_Y - (Overlap>>1)); y++)
		WriteSubbands(Data, DSize_Initial_X, DSize_Current_X, TData, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, &y, Last_Level);	
}

inline __device__ void WriteSubbands_Bottom(DATATYPE* Data, int DSize_Current_X,  int DSize_Initial_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int Last_Level, int Overlap)
{	
	for(int y = 0; y < (Overlap>>1); y+=2)

		UpdateSubbandsCoordinates_Scheduler(DSize_Current_X, DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, Last_Level);

	for(int y = (Overlap>>1); y < TDSize_Y; y++)

			WriteSubbands(Data, DSize_Initial_X, DSize_Current_X, TData, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, &y, Last_Level);
}

inline __device__ void WriteSubbands_Scheduler(DATATYPE* Data, int DSize_Current_X,  int DSize_Initial_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int Last_Level, int Incorrect_Vertical_Top, int Incorrect_Vertical_Bottom, int Overlap)
{
	if(			Incorrect_Vertical_Top == 0)		WriteSubbands_Top(Data, DSize_Current_X, DSize_Initial_X, TData, TDSize_Y, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, Last_Level, Overlap);			
	else if(	Incorrect_Vertical_Bottom == 0)		WriteSubbands_Bottom(Data, DSize_Current_X, DSize_Initial_X, TData, TDSize_Y, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, Last_Level, Overlap);	
	else											WriteSubbands_Middle(Data, DSize_Current_X, DSize_Initial_X, TData, TDSize_Y, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, Last_Level, Overlap);

}

inline __device__ void ReadSubbands_iteration(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH, int* index){

	TData[((*index)<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data[*TCoordinate_HL];

	++(*index);

	TData[((*index)<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data[*TCoordinate_LH];
	TData[((*index)<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data[*TCoordinate_LH + (DSize_Current_X>>1)];
}

inline __device__ void ReadSubbands(DATATYPE* Data, int DSize_Current_X, int DSize_Initial_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH)
{		
	for(int y = 0; y < TDSize_Y; y++){
		TData[(y<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data[*TCoordinate_LL];

		ReadSubbands_iteration(Data, DSize_Current_X, TData, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, &y);		
		UpdateSubbandsCoordinates(DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH);
	}		
}

inline __device__ void ReadSubbands_LLaux(DATATYPE* Data, DATATYPE* Data_LL, int DSize_Current_X, int DSize_Initial_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate_LL, int* TCoordinate_HL, int* TCoordinate_LH)
{		
	for(int y = 0; y < TDSize_Y; y++){
		TData[(y<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = Data_LL[*TCoordinate_LL];
		
		ReadSubbands_iteration(Data, DSize_Current_X, TData, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH, &y);		
		UpdateSubbandsCoordinates_LLaux(DSize_Current_X, DSize_Initial_X, TCoordinate_LL, TCoordinate_HL, TCoordinate_LH);
	}		
}

inline __device__ void WriteBlock_int1(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int* index)
{	
	Data[*TCoordinate] = TData[((*index)<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)];
	Data[*TCoordinate+1] = TData[(((*index)<<1)+1)+(SHARED_MEMORY_STRIDE*threadIdx.x)];

	*TCoordinate += DSize_Current_X;
}

__device__ __forceinline__ void st2(DATATYPE2* a, DATATYPE b, DATATYPE c)
{

	#if DWT53_or_DWT97 == 1
		#if DATATYPE_16BITS_or_32BITS == 1
			asm ("st.global.wt.v2.u16 [%0], {%1,%2};" :: "l"(a) , "h"(b), "h"(c));
		#else
			asm ("st.global.wt.v2.u32 [%0], {%1,%2};" :: "l"(a) , "r"(b), "r"(c));
		#endif
	#else
		#if DATATYPE_16BITS_or_32BITS == 1
			asm ("st.global.wt.v2.u16 [%0], {%1,%2};" :: "l"(a) , "h"(b), "h"(c));
		#else
			asm ("st.global.wt.v2.f32 [%0], {%1,%2};" :: "l"(a) , "f"(b), "f"(c));
		#endif
	#endif
}


inline __device__ void WriteBlock_int2(DATATYPE2* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int* index)
{			

	st2(Data+(*TCoordinate), TData[((*index)*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)], TData[((*index)*2) +1+(SHARED_MEMORY_STRIDE*threadIdx.x)]);
	//Data[*TCoordinate] = TData[((*index)*2)+(SHARED_MEMORY_STRIDE*threadIdx.x)];
	*TCoordinate += DSize_Current_X;

}


inline __device__ void WriteBlock(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int* index)
{	
	//WriteBlock_int1(Data, DSize_Current_X, TData, TDSize_Y, TCoordinate, index);
	WriteBlock_int2((DATATYPE2*)Data, DSize_Current_X, TData, TDSize_Y, TCoordinate, index);
}


inline __device__ void WriteBlock_Top(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int Overlap)
{
	for(int y = 0; y < (TDSize_Y - (Overlap>>1)); y++)
		WriteBlock(Data, DSize_Current_X, TData, TDSize_Y, TCoordinate, &y);
}

inline __device__ void WriteBlock_Middle(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int Overlap)
{
	for(int y = 0; y < (Overlap>>1); y++)
		*TCoordinate += DSize_Current_X;
	
	for(int y = (Overlap>>1); y < (TDSize_Y - (Overlap>>1)); y++)
		WriteBlock(Data, DSize_Current_X, TData, TDSize_Y, TCoordinate, &y);
}

inline __device__ void WriteBlock_Bottom(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int Overlap)
{	
	for(int y = 0; y < (Overlap>>1); y++)
		*TCoordinate += DSize_Current_X;

	for(int y = (Overlap>>1); y < TDSize_Y; y++)
		WriteBlock(Data, DSize_Current_X, TData, TDSize_Y, TCoordinate, &y);
}


inline __device__ void WriteBlock_Scheduler(DATATYPE* Data, int DSize_Current_X, REG_DATATYPE* TData, int TDSize_Y, int* TCoordinate, int Incorrect_Vertical_Top, int Incorrect_Vertical_Bottom, int Overlap)
{
	*(TCoordinate)>>=1;

	if(			Incorrect_Vertical_Top == 0)		WriteBlock_Top(Data, DSize_Current_X/2, TData, TDSize_Y, TCoordinate, Overlap);
	else if(	Incorrect_Vertical_Bottom == 0)		WriteBlock_Bottom(Data, DSize_Current_X/2, TData, TDSize_Y, TCoordinate, Overlap);
	else											WriteBlock_Middle(Data, DSize_Current_X/2, TData, TDSize_Y, TCoordinate, Overlap);
	
	*(TCoordinate)<<=1;
}

//END - <DEVICE> DATA MANAGEMENT FUNCTIONS -----------------------------------------------------------------------














//START - <DEVICE> PRE-COMPUTE FUNCTIONS -----------------------------------------------------------------------

//Assign a data block to a warp (compute the coordinates from where the warp will fetch its data)
inline __device__ void InitializeCoordinates(	int DSize_Current_X, int DSize_Initial_X, int DSize_Current_Y, int* TCoordinate_X, int* TCoordinate_Y, int* TCoordinate, int* TCoordinate_LL, 
												int* TCoordinate_HL, int* TCoordinate_LH, int LaneID, int WarpID, int NWarps_X, int NWarps_Y, int WarpWork_Y, int LL_offset, 
												int Special_Level, int Overlap)
{
	int X_effective_work = ((WARPSIZE * NELEMENTS_THREAD_X)-Overlap);
	int Y_effective_work = WarpWork_Y;	
	
	int X_border_coordinate_correction = (((WarpID+1) % NWarps_X)==0)?1:0;
	int Y_border_coordinate_correction = (WarpID> ((NWarps_Y*NWarps_X) - NWarps_X - 1))?1:0;	

	*TCoordinate_X =		(((WarpID % NWarps_X) * X_effective_work) + (LaneID * NELEMENTS_THREAD_X));

	if(X_border_coordinate_correction) 		*TCoordinate_X -= (X_effective_work - (DSize_Current_X % X_effective_work))%X_effective_work + Overlap;

	*TCoordinate_Y =		((WarpID/NWarps_X) * (Y_effective_work));

	if(Y_border_coordinate_correction)		*TCoordinate_Y -= (Y_effective_work - ((DSize_Current_Y- Overlap) % Y_effective_work)) % Y_effective_work ;	

	*TCoordinate =			DSize_Current_X*(*TCoordinate_Y) + *TCoordinate_X;

	if(Special_Level==1) 	*TCoordinate_LL = 		((*TCoordinate_Y>>1)*DSize_Initial_X) + (*TCoordinate_X>>1) ;
	else					*TCoordinate_LL = 		((*TCoordinate_Y>>1)*(DSize_Current_X>>1)) + (*TCoordinate_X>>1) + LL_offset;

	*TCoordinate_HL = 		((*TCoordinate_Y>>1)*DSize_Initial_X) + (*TCoordinate_X>>1) + (DSize_Current_X>>1);
	*TCoordinate_LH = 		(((*TCoordinate_Y>>1) + (DSize_Current_Y>>1))*DSize_Initial_X) + (*TCoordinate_X>>1);

}


//With some image and data block sizes some warps can be assigned to data blocks beyond the image borders. This function check if this happens, and its output will be used in the time to write back the results of the DWT
inline __device__ void IncorrectBorderValues(	int LaneID, int WarpID, int NWarps_X, int NWarps_Y, int* Incorrect_Horizontal, int* Incorrect_Vertical_Top, int* Incorrect_Vertical_Bottom, int Overlap)
{
	if(		(((WarpID % NWarps_X)!=0)		&&	(LaneID <((Overlap>>1)/NELEMENTS_THREAD_X))) ||
			(((WarpID + 1) % NWarps_X)!=0)	&&	(LaneID >(WARPSIZE -1 - ((Overlap>>1)/NELEMENTS_THREAD_X))))
				
				*Incorrect_Horizontal = 1;

	if(		WarpID > (NWarps_X - 1))

				*Incorrect_Vertical_Top = 1;

	if(		WarpID < (NWarps_X*(NWarps_Y-1)))
				
				*Incorrect_Vertical_Bottom = 1;
}

//END - <DEVICE> PRE-COMPUTE FUNCTIONS -----------------------------------------------------------------------














//START - <DEVICE> CUDA KERNELS -----------------------------------------------------------------------

//CUDA KERNEL that computes the forward DWT over an input image. The same kernel is used for both for the 5/3 and 9/7 DWT
__global__ void Kernel_DWT_F(		
									DATATYPE* device_original_image, 
									DATATYPE* device_result_image,
									int DSize_Current_X,
									int DSize_Initial_X,
									int DSize_Current_Y,
									int	NWarps_X, 
									int	NWarps_Y, 
									int WarpWork_Y, 
									int NWarps_Block,
									int Write_LL_offset,
									int Last_Level,
									int write
								)
{
	extern __shared__ int synthetic_shared_memory[];
	
	#if FULLSHARED != 1
		register REG_DATATYPE TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];		
	#else
		__shared__ REG_DATATYPE TData[SHARED_MEMORY_STRIDE*NTHREADSBLOCK_DWT_F];
	#endif

	#if SHUFFLE != 1
		__shared__ REG_DATATYPE Shared_Data[NTHREADSBLOCK_DWT_F];
	#endif

	int LaneID = 			threadIdx.x & 0x1f;
	int WarpID = 			(((threadIdx.x >> 5) + (blockIdx.x * NWarps_Block)));	
	int Idle_Warp = 		0;
	int TCoordinate_X = 	0;	
	int TCoordinate_Y = 	0;
	int TCoordinate = 		0;
	int TCoordinate_LL =	0;
	int TCoordinate_HL = 	0;
	int TCoordinate_LH = 	0;		

	int Incorrect_Horizontal =	0;
	int Incorrect_Vertical_Top = 0;
	int Incorrect_Vertical_Bottom = 0;

	
	Idle_Warp = 		(WarpID < (NWarps_X*NWarps_Y))?0:1;

	if(Idle_Warp) return;
	
	InitializeCoordinates(	DSize_Current_X, DSize_Initial_X, DSize_Current_Y, &TCoordinate_X, &TCoordinate_Y, &TCoordinate, &TCoordinate_LL, &TCoordinate_HL, &TCoordinate_LH, 
							LaneID, WarpID, NWarps_X, NWarps_Y, WarpWork_Y, Write_LL_offset, Last_Level, OVERLAP);	

	IncorrectBorderValues(	LaneID, WarpID, NWarps_X, NWarps_Y, &Incorrect_Horizontal, &Incorrect_Vertical_Top, &Incorrect_Vertical_Bottom, OVERLAP);

	#if READ == 1				
		ReadBlock2((DATATYPE2*)device_original_image, DSize_Current_X/2, TData, NELEMENTS_THREAD_Y, &TCoordinate);

	#else 
		for(int y = 0; y < NELEMENTS_THREAD_Y; y++){

			TData[(y<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = threadIdx.x + y;
			TData[(y<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = threadIdx.x + y +1;}
	#endif

	#if VERTICAL_COMPUTATION == 1
		#if DWT53_or_DWT97 == 1 	
			Vertical_Filter_Forward_53(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
		#else 
			Vertical_Filter_Forward_97(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
		#endif
	#endif
	#if HORIZONTAL_COMPUTATION == 1
		#if FULLSHARED == 1
			#if DWT53_or_DWT97 == 1 	
					Horizontal_Filter_Forward_53_Full_Shared((VOLATILE REG_DATATYPE*)TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);	
				#else 
					Horizontal_Filter_Forward_97_Full_Shared((VOLATILE REG_DATATYPE*)TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#endif		
		#else
			#if SHUFFLE == 1			

				#if DWT53_or_DWT97 == 1 	
					Horizontal_Filter_Forward_53_Shuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#else 
					Horizontal_Filter_Forward_97_Shuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#endif	
			#else				
				#if DWT53_or_DWT97 == 1 	
					Horizontal_Filter_Forward_53_Shared(TData, Shared_Data, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#else 
					Horizontal_Filter_Forward_97_Shared(TData, Shared_Data, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#endif
			#endif
		#endif
	#endif
	
	if(write)			
		if(Incorrect_Horizontal==0)
			WriteSubbands_Scheduler(device_result_image, DSize_Current_X, DSize_Initial_X, TData, NELEMENTS_THREAD_Y, &TCoordinate_LL, &TCoordinate_HL, &TCoordinate_LH, Last_Level, Incorrect_Vertical_Top, Incorrect_Vertical_Bottom, OVERLAP);			
	
}

//CUDA KERNEL that computes the reverse DWT over an input image. The same kernel is used for both for the 5/3 and 9/7 DWT
__global__ void Kernel_DWT_R(	
									DATATYPE* device_original_image, 
									DATATYPE* device_result_image,
									int DSize_Current_X,
									int DSize_Initial_X,
									int DSize_Current_Y,
									int	NWarps_X, 
									int	NWarps_Y,
									int WarpWork_Y, 
									int NWarps_Block,
									int Read_LL_offset,
									int Write_offset,
									int First_Level,
									int write
								)
{

	extern __shared__ int synthetic_shared_memory[];
	
	#if FULLSHARED != 1
		register REG_DATATYPE TData[NELEMENTS_THREAD_Y*NELEMENTS_THREAD_X];		
	#else
		__shared__ REG_DATATYPE TData[SHARED_MEMORY_STRIDE*NTHREADSBLOCK_DWT_F];
	#endif

	#if SHUFFLE != 1
		__shared__ REG_DATATYPE Shared_Data[NTHREADSBLOCK_DWT_F];
	#endif

	int LaneID = 			threadIdx.x & 0x1f;
	int WarpID = 			(((threadIdx.x >> 5) + (blockIdx.x * NWarps_Block)));
	int TCoordinate_X = 	0;	
	int TCoordinate_Y = 	0;
	int TCoordinate = 		0;
	int TCoordinate_LL = 	0;
	int TCoordinate_HL = 	0;
	int TCoordinate_LH = 	0;	

	int Incorrect_Horizontal =	0;
	int Incorrect_Vertical_Top = 0;
	int Incorrect_Vertical_Bottom = 0;
	int Idle_Warp = 		(WarpID < (NWarps_X*NWarps_Y))?0:1;

	if(Idle_Warp) return;
		
	InitializeCoordinates(	DSize_Current_X, DSize_Initial_X, DSize_Current_Y, &TCoordinate_X, &TCoordinate_Y, &TCoordinate, &TCoordinate_LL, &TCoordinate_HL, &TCoordinate_LH, 
							LaneID, WarpID, NWarps_X, NWarps_Y, WarpWork_Y, Read_LL_offset, First_Level, OVERLAP);	
	
	IncorrectBorderValues(	LaneID, WarpID, NWarps_X, NWarps_Y, &Incorrect_Horizontal, &Incorrect_Vertical_Top, &Incorrect_Vertical_Bottom, OVERLAP);	

	#if READ == 1

		if(First_Level)		ReadSubbands(device_original_image, DSize_Current_X, DSize_Initial_X, TData, NELEMENTS_THREAD_Y, &TCoordinate_LL, &TCoordinate_HL, &TCoordinate_LH);
		else 				ReadSubbands_LLaux(device_original_image, device_result_image-Write_offset, DSize_Current_X, DSize_Initial_X, TData, NELEMENTS_THREAD_Y, &TCoordinate_LL, &TCoordinate_HL, &TCoordinate_LH);

	#else 
		for(int y = 0; y < NELEMENTS_THREAD_Y; y++){

			TData[(y<<1)+(SHARED_MEMORY_STRIDE*threadIdx.x)] = threadIdx.x + y;
			TData[(y<<1)+1+(SHARED_MEMORY_STRIDE*threadIdx.x)] = threadIdx.x + y +1;}
	#endif

	#if HORIZONTAL_COMPUTATION == 1
		#if FULLSHARED == 1
			#if DWT53_or_DWT97 == 1 	
				Horizontal_Filter_Reverse_53_Full_Shared((VOLATILE REG_DATATYPE*)TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
			#else 
				Horizontal_Filter_Reverse_97_Full_Shared((VOLATILE REG_DATATYPE*)TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
			#endif
		#else 
			#if SHUFFLE == 1
				#if DWT53_or_DWT97 == 1 	
					Horizontal_Filter_Reverse_53_Shuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#else 
					Horizontal_Filter_Reverse_97_Shuffle(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#endif
			#else
				#if DWT53_or_DWT97 == 1 	
					Horizontal_Filter_Reverse_53_Shared(TData, Shared_Data, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#else 
					Horizontal_Filter_Reverse_97_Shared(TData, Shared_Data, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
				#endif
			#endif
		#endif
	#endif
	#if VERTICAL_COMPUTATION == 1
		#if DWT53_or_DWT97 == 1 	
			Vertical_Filter_Reverse_53(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
		#else 
			Vertical_Filter_Reverse_97(TData, NELEMENTS_THREAD_Y, NELEMENTS_THREAD_X);
		#endif
	#endif

	if(write)			
		if(Incorrect_Horizontal==0)
			WriteBlock_Scheduler(device_result_image, DSize_Current_X, TData, NELEMENTS_THREAD_Y, &TCoordinate, Incorrect_Vertical_Top, Incorrect_Vertical_Bottom, OVERLAP);	
}

//END - <DEVICE> CUDA KERNELS -----------------------------------------------------------------------














//START - <HOST> DWT FUNCTIONS -----------------------------------------------------------------------

//DWT FORWARD

static inline void DWT_F(int DWT_Levels, int DSize_Initial_X, int DSize_Initial_Y, DATATYPE* DData_Initial, DATATYPE* DData_Final){

	int 		Warps_Row, Warps_Column, warp_vertical_length_work, CUDA_number_blocks, CUDA_number_warps;
	int 		Write_LL_offset = 0;
	int			Last_Level = 0;
	int 		DSize_Current_X = DSize_Initial_X;
	int 		DSize_Current_Y = DSize_Initial_Y;
	DATATYPE*	DData_Initial_aux =	DData_Initial;

//A CUDA kernel is launched for every DWT level
	for(int Current_Level = DWT_Levels; Current_Level>0; --Current_Level)
	{		
		Warps_Row = 					(int)ceil(DSize_Current_X/((float)(WARPSIZE*NELEMENTS_THREAD_X)-OVERLAP));	
		warp_vertical_length_work = 	(NELEMENTS_THREAD_Y)-OVERLAP;
		Warps_Column = 					(int)ceil((((DSize_Current_Y - OVERLAP)/(float)(warp_vertical_length_work))));
		CUDA_number_warps = 			Warps_Row *	Warps_Column;					
		CUDA_number_blocks = 			(int)ceil((CUDA_number_warps*WARPSIZE)/(float)(NTHREADSBLOCK_DWT_F));
		
		Last_Level = (Current_Level == 1) ? 1 : Last_Level;
		Write_LL_offset += (DSize_Current_X*DSize_Current_Y);

		Kernel_DWT_F<<<CUDA_number_blocks,NTHREADSBLOCK_DWT_F, SYNTHETIC_SHARED>>>
									(	
										DData_Initial_aux, 
										DData_Final,									
										DSize_Current_X,
										DSize_Initial_X,
										DSize_Current_Y, 
										Warps_Row, 
										Warps_Column,
										warp_vertical_length_work, 
										NTHREADSBLOCK_DWT_F/WARPSIZE,
										Write_LL_offset, 
										Last_Level,
										WRITE
									);
		
		DData_Initial_aux = DData_Final + (Write_LL_offset);
		
		DSize_Current_X>>=1;
		DSize_Current_Y>>=1;
	}
}

//DWT REVERSE

static inline void DWT_R(int DWT_Levels, int DSize_Initial_X, int DSize_Initial_Y, DATATYPE* DData_Initial, DATATYPE* DData_Final){

	int 		Warps_Row, Warps_Column, warp_vertical_length_work, CUDA_number_blocks, CUDA_number_warps;
	int 		Read_LL_offset = 0;
	int 		Write_offset = 0;
	int			First_Level = 1;
	int 		DSize_Current_X = DSize_Initial_X>>(DWT_Levels-1);
	int 		DSize_Current_Y = DSize_Initial_Y>>(DWT_Levels-1);
	DATATYPE*		DData_Final_aux =	DData_Final;

//A CUDA kernel is launched for every DWT level
	for(int Current_Level = DWT_Levels; Current_Level>0; --Current_Level)
	{
		Warps_Row = 					(int)ceil(DSize_Current_X/((float)(WARPSIZE*NELEMENTS_THREAD_X)-OVERLAP));
		warp_vertical_length_work = 	(NELEMENTS_THREAD_Y)-OVERLAP;
		Warps_Column = 					(int)ceil(((DSize_Current_Y - OVERLAP)/(float)(warp_vertical_length_work)));
		CUDA_number_warps = 			Warps_Row *	Warps_Column;					
		CUDA_number_blocks = 			(int)ceil((CUDA_number_warps*WARPSIZE)/(float)NTHREADSBLOCK_DWT_R);

		Kernel_DWT_R<<<CUDA_number_blocks,NTHREADSBLOCK_DWT_R, SYNTHETIC_SHARED>>>
									(	
										DData_Initial,
										DData_Final_aux,
										DSize_Current_X,
										DSize_Initial_X,
										DSize_Current_Y, 
										Warps_Row, 
										Warps_Column,
										warp_vertical_length_work, 
										NTHREADSBLOCK_DWT_R/WARPSIZE,
										Read_LL_offset,
										Write_offset,
										First_Level,
										WRITE				
									);
		First_Level = 0;		

		Read_LL_offset += (Write_offset - Read_LL_offset);
		Write_offset += (DSize_Current_X*DSize_Current_Y);
		DData_Final_aux = DData_Final + Write_offset;

		DSize_Current_X<<=1;
		DSize_Current_Y<<=1;
	}
}

//END - <HOST> DWT FUNCTIONS -----------------------------------------------------------------------














//START - <HOST> PRE/POST COMPUTE FUNCTIONS -----------------------------------------------------------------------

static inline void Kernel_launcher(int DWT_Direction, int DWT_Levels, int DSize_X, int DSize_Y, DATATYPE* DData_Initial, DATATYPE* DData_Final){
	switch(DWT_Direction){
		case FORWARD:	
			DWT_F(DWT_Levels, DSize_X, DSize_Y, (DATATYPE*) DData_Initial, (DATATYPE*) DData_Final);	
		break;

		case REVERSE:
			DWT_R(DWT_Levels, DSize_X, DSize_Y, DData_Initial, DData_Final);
		break;
	}
}

static inline void Device_memory_allocator(int DWT_Direction, int DWT_Levels, DATATYPE* HData, int HDSize_X, int HDSize_Y, DATATYPE** DData_Initial, DATATYPE** DData_Final, int* DData_Extra){
		
	size_t 		DSize  = HDSize_X*HDSize_Y*sizeof(DATATYPE);
		
	*DData_Extra = 0;	

	for(int i=1; i<DWT_Levels; ++i) *DData_Extra +=  (HDSize_X/(2<<(i-1)))* (HDSize_Y/(2<<(i-1)));	

	cudaMalloc ((void**) &(*DData_Initial), DSize);
	cudaMalloc ((void**) &(*DData_Final), DSize + ((*DData_Extra)* sizeof(DATATYPE)));

	cudaMemcpy(*DData_Initial, (DATATYPE*)HData, DSize, cudaMemcpyHostToDevice);
}

static inline void Device_memory_deallocator(int DWT_Direction, DATATYPE* HData, int HDSize_X, int HDSize_Y, DATATYPE* DData_Initial, DATATYPE* DData_Final, int DData_Extra){
	
	size_t 		DSize = HDSize_X*HDSize_Y*sizeof(DATATYPE);
	
	if( (DWT_Direction==REVERSE) )		cudaMemcpy(HData, DData_Final + DData_Extra, DSize, cudaMemcpyDeviceToHost);
	else 								cudaMemcpy(HData, DData_Final, DSize, cudaMemcpyDeviceToHost);

	cudaFree(DData_Initial);
	cudaFree(DData_Final);
	
}


//Apply DWT_Levels over an input data HData of sizes HDSize_X x HDSize_Y. DWT_Direction determines wether it computes the forward or the reverse DWT.
void CUDA_DWT(int DWT_Direction,int DWT_Levels, DATATYPE* HData, int HDSize_X, int HDSize_Y)
{
			
	DATATYPE* 		DData_Initial; 
	DATATYPE* 		DData_Final;
	int				DData_Extra;	
#if FULLSHARED == 1
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#else
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif
	Device_memory_allocator(DWT_Direction, DWT_Levels, HData, HDSize_X, HDSize_Y, &DData_Initial, &DData_Final, &DData_Extra);
	Kernel_launcher(DWT_Direction, DWT_Levels, HDSize_X, HDSize_Y, DData_Initial, DData_Final);
	Device_memory_deallocator(DWT_Direction, HData, HDSize_X, HDSize_Y, DData_Initial, DData_Final, DData_Extra);
}	

//END - <HOST> PRE/POST COMPUTE FUNCTIONS -----------------------------------------------------------------------













	
