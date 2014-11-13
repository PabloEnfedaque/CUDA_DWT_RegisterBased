#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "RBased_DWT_common.h"


#if DATATYPE_16BITS_or_32BITS == 1
	#if DWT53_or_DWT97 == 0
		#define TESTDATATYPE float
	#else 		
		#define TESTDATATYPE DATATYPE
	#endif
#else 
		#define TESTDATATYPE DATATYPE
#endif


void CUDA_DWT(int, int, DATATYPE* , int , int);


void Check(TESTDATATYPE* image1, DATATYPE* image2, int xSize, int ySize, long int *correct_flag, long int *wrong_count){

	for(int i=0;(i<ySize);i++){	
		for(int j=0;(j<xSize);j++) {


		#if DWT53_or_DWT97 == 1
			if(((DATATYPE)image1[(xSize*i)+j]) != ((int)image2[(xSize*i)+j])){
				
		#else 

			#if DATATYPE_16BITS_or_32BITS == 1

			if(abs(((DATATYPE)(image1[(xSize*i)+j])) - (image2[(xSize*i)+j]))>16){			

			#else 
				
			if(abs(((DATATYPE)(image1[(xSize*i)+j])) - (image2[(xSize*i)+j]))>1){			
	
			#endif

		#endif

				if(*correct_flag==-1){
					*correct_flag = (xSize*i)+j;
					*wrong_count= *wrong_count +1 ;
				}
				else ++(*wrong_count);
			}
		}
	}
	
}


void Print_Results(int xSize,int ySize,long int correct_flag, long int wrong_count, int levels){

	if(correct_flag==-1)	printf("TEST	image size: %dx%d	with %d levels	>>>	OK\n", xSize, ySize, levels);
	else printf("TEST	image size: %dx%d	with %d levels	>>>	ERROR ...... first error position:  row %ld, column %ld //  number mismatches: %ld\n", 
					xSize, ySize, levels, correct_flag/xSize, correct_flag%xSize, wrong_count);
}

int main(int argc, char** argv)
{
	int aux;
	int ySize, xSize;
	int levels = LEVELS;

	printf("\nComputing the dwt...\n\n");		

		printf("------------------------------------------------------------------------------------\n");	
		#if DWT53_or_DWT97 == 1
			printf("				DWT 53\n");	
		#else 
			printf("				DWT 97\n");
		#endif
		printf("------------------------------------------------------------------------------------\n");
		printf("*** Comparing forward versus reverse DWT results, with some random generated samples ***\n");
		printf("------------------------------------------------------------------------------------\n\n");

	for(int i= 0;i<NEXPERIMNETS;i++)
	{	

		aux= EXPERIMENT_INI;
		ySize = ((i)*EXPERIMENT_INCREMENT)+EXPERIMENT_INI;
		xSize = ((i)*EXPERIMENT_INCREMENT)+EXPERIMENT_INI;
		
		srand(time(NULL));


		TESTDATATYPE *image1 = (TESTDATATYPE*)malloc((ySize) * (xSize) * sizeof(TESTDATATYPE));
		DATATYPE *image2 = (DATATYPE*)malloc((ySize) * (xSize) * sizeof(DATATYPE));


		for(int i=0;i<xSize;i++){	for(int j=0;j<ySize;j++){ 
			
			aux = (rand() % 255) -128; 

			image1[(xSize*i)+j]= aux; 
			image2[(xSize*i)+j]= aux; 

		}}		

		long int correct_flag=-1;
		long int wrong_count=0;

		CUDA_DWT(FORWARD, levels, image2 , xSize , ySize);
		CUDA_DWT(REVERSE, levels, image2 , xSize , ySize);		

		Check(image1, image2, xSize, ySize, &correct_flag, &wrong_count);
		Print_Results(xSize, ySize, correct_flag, wrong_count, levels);

				
		free(image1);
		free(image2);				
		
	}

	printf("\n");

	
	cudaDeviceReset(); 
	return(0);
}
