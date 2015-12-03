#ifndef SMOOTH_H
#define SMOOTH_H

#define BLOCK_SIZE	5
#define __STEP__ (BLOCK_SIZE/2)


#ifdef __CUDACC__
#define SMOOTH_CALL_LEVEL	__device__
#else
#define SMOOTH_CALL_LEVEL
#endif


/**
 * Simples soma de tamanhos
 * @param  cols       colunas totais da img
 * @param  pixel_size tamanho do pixel
 * @return            tamanho da linha
 */
SMOOTH_CALL_LEVEL
int getLineSize(int cols, int pixel_size);


/**
 * Aritmetica para o indice do ponteiro
 * @param  line       y > qual a linha
 * @param  col        x > qual a coluna
 * @param  col_size   tamanho dessa coluna (total)
 * @param  pixel_size tamanho do pixel
 * @return            indice (nao checa veracidade do indice)
 */
SMOOTH_CALL_LEVEL
int getIndex(int line, int col, int col_size, int pixel_size);


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
void getAverage(unsigned char* img, int x, int y, unsigned char* result, int pixel_size, int col_limit, int row_limit);

#endif