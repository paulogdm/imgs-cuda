#ifndef SMOOTH_H
#define SMOOTH_H

#define BLOCK_SIZE	5
#define __STEP__ (BLOCK_SIZE/2)


int getLineSize(int cols, int pixel_size);
int getIndex(int line, int col, int col_size, int pixel_size);
void getAverage(Image* img, int x, int y, unsigned char* result);
void smoothImage(Image *out, Image *in, int in_start, int in_end);
void smoothImage(Image *out, Image *in);

#endif