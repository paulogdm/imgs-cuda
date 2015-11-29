#ifndef SMOOTH_H
#define SMOOTH_H

#define BLOCK_SIZE	5
#define __STEP__ (BLOCK_SIZE/2)


#ifdef __CUDACC__
#define CUDA_CALL __host__ __device__
#else
#define CUDA_CALL
#endif

CUDA_CALL
int getLineSize(int cols, int pixel_size);
CUDA_CALL
int getIndex(int line, int col, int col_size, int pixel_size);
CUDA_CALL
void getAverage(unsigned char* pixel_array, int x, int y, unsigned char* result, int col_limit, int row_limit, int pixel_size);

#endif