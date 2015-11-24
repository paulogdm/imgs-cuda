#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <omp.h>
#include <image.h>

#include "smooth.h"


int getLineSize(int cols, int pixel_size){
	return pixel_size*cols*sizeof(unsigned char);
}

int getIndex(int line, int col, int col_size, int pixel_size){
	return ((line*col_size*pixel_size) + col*pixel_size)*sizeof(unsigned char);
}

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


void smoothImage(Image *out, Image *in, int in_start, int in_end){

	int pixelSize = in->getPixelSize();
	unsigned char* pixel_array = out->getData();
		
//	#pragma omp parallel
//	{
		unsigned char *buffer = (unsigned char*) calloc(pixelSize, sizeof(unsigned char));
		int index;

		#pragma omp for
		for (int i = in_start; i < in_end; i++){
				
			for (int j = 0; j < in->getCols(); j++){

				getAverage(in, i, j, buffer);
				index = getIndex(i-in_start, j, in->getCols(), pixelSize);
				
				for(int c=0; c < pixelSize; c++){
					pixel_array[c + index] = buffer[c];
				}			
			}
		}
	
		free(buffer); //liberando buffer
//	}
}


void smoothImage(Image *out, Image *in){

	int pixelSize = in->getPixelSize();
	unsigned char *buffer = (unsigned char*) calloc(pixelSize, sizeof(unsigned char));
	unsigned char *pixel_array = out->getData();
	int index;

	for (int i = 0; i < in->getRows(); i++){
		for (int j = 0; j < in->getCols(); j++){

			getAverage(in, i, j, buffer);
			index = getIndex(i, j, in->getCols(), pixelSize);

			for(int c = 0; c < pixelSize; c++){
				pixel_array[c + index] = buffer[c];
			}			
		}
	}

	free(buffer); //liberando buffer
}

