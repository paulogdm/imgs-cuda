#ifndef SMOOTH_H
#define SMOOTH_H

#define BLOCK_SIZE	5
#define __STEP__ (BLOCK_SIZE/2)


#ifdef __CUDACC__
#define SMOOTH_CALL_LEVEL	__device__
#else
#define SMOOTH_CALL_LEVEL
#endif

SMOOTH_CALL_LEVEL
int getLineSize(int cols, int pixel_size);

SMOOTH_CALL_LEVEL
int getIndex(int line, int col, int col_size, int pixel_size);

SMOOTH_CALL_LEVEL
void getAverage(unsigned char* img, int x, int y, unsigned char* result, int pixel_size, int col_limit, int row_limit);

#endif