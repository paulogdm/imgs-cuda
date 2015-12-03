#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <image.h>

#include "smooth.h"


/**
 * Simples soma de tamanhos
 * @param  cols       colunas totais da img
 * @param  pixel_size tamanho do pixel
 * @return            tamanho da linha
 */
SMOOTH_CALL_LEVEL
int getLineSize(int cols, int pixel_size){
	return pixel_size*cols*sizeof(unsigned char);
}

/**
 * Aritmetica para o indice do ponteiro
 * @param  line       y > qual a linha
 * @param  col        x > qual a coluna
 * @param  col_size   tamanho dessa coluna (total)
 * @param  pixel_size tamanho do pixel
 * @return            indice (nao checa veracidade do indice)
 */
SMOOTH_CALL_LEVEL
int getIndex(int line, int col, int col_size, int pixel_size){
	return ((line*col_size*pixel_size) + col*pixel_size)*sizeof(unsigned char);
}

/**
 * Pega a media dos pixeis baseada no tamanho do bloco
 * @param pixel_array array de pixels.
 * @param x           coordenada x
 * @param y           coordenada y
 * @param result      resultado
 * @param pixel_size  tamanho do pixel
 * @param col_limit   tamanho max da coluna
 * @param row_limit   tamanho max da linha
 */
SMOOTH_CALL_LEVEL
void getAverage(unsigned char* pixel_array, int x, int y, unsigned char* result, 
				int pixel_size, int col_limit, int row_limit){

	int counter=0; //contador de pixeis validos
	int color[3]={0,0,0}; //buffer de contagem

	//percorrendo bloco
	for(int i = x - __STEP__; i <= x + __STEP__; i++){
		for(int j = y - __STEP__; j <= y + __STEP__; j++){
			if(i < 0 || i >= row_limit || j < 0 || j >= col_limit){
				//se algo precisa ser feito em caso de ultrapassar margens
			} else {
				//pixel valido sendo contabilizado
				counter++; //quantos pixels fazem parte da media
				int index = getIndex(i, j, col_limit, pixel_size);
				for(int c = 0; c < pixel_size; c++){
					color[c] += pixel_array[index+c];
				}
			}
		}
	}
	
	//pragma aqui melhorou a performance em 8%
	#pragma unroll
	for(int i = 0; i < pixel_size; i++){
		result[i] = (color[i] / counter);
	}
}





