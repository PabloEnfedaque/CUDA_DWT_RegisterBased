The code for this project has been developed for research pourposes. 
Square images of size i\*2 x i\*2 with i > 64 have been employed for all experiments. This implementation may not properly work for other image sizes.

The "main_example.cu" first applies the Forward DWT and then the Reverse DWT over a set of randomly generated images. It can be compiled, executed and profiled using the nvcc compiler as follows:

	$nvcc -c -O3 -arch=sm_35 -use_fast_math -Xptxas -v -DDWT53_or_DWT97=1 -DNEXPERIMENTS=1 -DEXPERIMENT_INI=1024 -DNELEMENTS_THREAD_Y=20 -DSHUFFLE=1 -DLEVELS=5 *.cu
	$nvcc *.o -o test
	$nvprof --print-gpu-trace ./test

Several precompilation parameters are defined in "RBased_DWT_common.h". Most important are:

-DDWT53_or_DWT97: =1 the DWT 5/3 is computed, =0 the DWT 9/7

-DNEXPERIMENTS: Number of randomly generated images computed. 

-DEXPERIMENT_INI: Size in both dimensions of the randomly generated input image employed.

-DNELEMENTS_THREAD_Y : Length in samples of the Data block computed by each warp. Width is fixed to 64 samples.

-DSHUFFLE: If =1 the shuffle instructions are employed in the computation. If =0, an intermediate buffer in shared memory is used instead (for pre-Kepler architectures).

-DLEVELS : Number of DWT decomposition levels applied.


Depending on the CUDA architecture, the code can be compiled as follows:

Fermi 2.0:

	$nvcc -c -O3 -arch=sm_20 -use_fast_math -Xptxas -v -DDWT53_or_DWT97=1 -DNEXPERIMNETS=1 -DEXPERIMENT_INI=1024 -DNELEMENTS_THREAD_Y=20 -DSHUFFLE=0 -DLEVELS=5 *.cu

Kepler 3.5:

	$nvcc -c -O3 -arch=sm_35 -use_fast_math -Xptxas -v -DDWT53_or_DWT97=1 -DNEXPERIMNETS=1 -DEXPERIMENT_INI=1024 -DNELEMENTS_THREAD_Y=20 -DSHUFFLE=1 -DLEVELS=5 *.cu

Maxwell 5.0:

	$nvcc -c -O3 -arch=sm_50 -use_fast_math -Xptxas -v -DDWT53_or_DWT97=1 -DNEXPERIMNETS=1 -DEXPERIMENT_INI=1024 -DNELEMENTS_THREAD_Y=20 -DSHUFFLE=1 -DLEVELS=5 *.cu
