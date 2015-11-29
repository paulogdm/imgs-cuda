#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <image.h>

#include "smooth.h"


CUDA_CALL
int getLineSize(int cols, int pixel_size){
	return pixel_size*cols*sizeof(unsigned char);
}

CUDA_CALL
int getIndex(int line, int col, int col_size, int pixel_size){
	return ((line*col_size*pixel_size) + col*pixel_size)*sizeof(unsigned char);
}


CUDA_CALL
void getAverage(Image* img, int x, int y, unsigned char* result){

	int counter=0;
	int color[3]={0,0,0};
	int index;

	for(int i = x - __STEP__; i <= x + __STEP__; i++){
		unsigned char* pixel_array = img->getData();

		for(int j = y - __STEP__; j <= y + __STEP__; j++){
			
			if(i < 0 || i >= img->getRows() || j < 0 || j >= img->getCols()){
				//se algo precisa ser feito em caso de ultrapassar margens
			} else {
				counter++; //quantos pixels fazem parte da media
				index = getIndex(i, j, img->getCols(), img->getPixelSize());

				for(int c = 0; c < img->getPixelSize(); c++){
					color[c] += pixel_array[index+c];
				}
			}
		}
	}
	

	for(int i = 0; i < img->getPixelSize(); i++){
		result[i] = (color[i] / counter);
	}
}





