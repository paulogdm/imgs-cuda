#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <image.h>

#include "smooth.h"

SMOOTH_CALL_LEVEL
int getLineSize(int cols, int pixel_size){
	return pixel_size*cols*sizeof(unsigned char);
}

SMOOTH_CALL_LEVEL
int getIndex(int line, int col, int col_size, int pixel_size){
	return ((line*col_size*pixel_size) + col*pixel_size)*sizeof(unsigned char);
}

SMOOTH_CALL_LEVEL
void getAverage(unsigned char* pixel_array, int x, int y, unsigned char* result, int pixel_size, int col_limit, int row_limit){

	int counter=0;
	int color[3]={0,0,0};

	for(int i = x - __STEP__; i <= x + __STEP__; i++){
		for(int j = y - __STEP__; j <= y + __STEP__; j++){

			if(i < 0 || i >= row_limit || j < 0 || j >= col_limit){
				//se algo precisa ser feito em caso de ultrapassar margens
			} else {
				counter++; //quantos pixels fazem parte da media
				int index = getIndex(i, j, col_limit, pixel_size);

				for(int c = 0; c < pixel_size; c++){
					color[c] += pixel_array[index+c];
				}
			}
		}
	}
	
	#pragma unroll
	for(int i = 0; i < pixel_size; i++){
		result[i] = (color[i] / counter);
	}
}





