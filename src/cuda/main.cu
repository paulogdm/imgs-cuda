#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <image.h>
#include <smooth.h>

Image* readImage(const char *name){

	Image *buffer = NULL;
	
	buffer=createImage(name);
	
	if(buffer != NULL)
		buffer->readFile(name);

	return buffer;
}

void writeImage(const char *name, Image *out){
	if(out != NULL)
		out->writeFile(name);
}

__device__ void smoothImage(Image *out, Image *in){

	int pixelSize = in->getPixelSize();
	unsigned char *buffer = (unsigned char*) calloc(pixelSize, sizeof(unsigned char));
	unsigned char *pixel_array = out->getData();

	int row_limit = in->getRows();
	int col_limit = in->getCols();

	for (int row = blockIdx.x; row < in->getRows(); row+=gridDim.x){
		for (int col = threadIdx.x; col <= in->getCols(); col+=blockDim.x){

			getAverage(in, i, j, buffer);
			int index = getIndex(i, j, in->getCols(), pixelSize);

			for(int c = 0; c < pixelSize; c++){
				pixel_array[c + index] = buffer[c];
			}			
		}
	}

	free(buffer); //liberando buffer
}

int main(int argc, const char **argv){

	Image *in;
	Image *out;

	if(argc != 3){
		printf("Usage: %s <IMAGE_IN> <IMAGE_OUT>\n", argv[0]);
		return 1;
	}

	in = readImage(argv[1]);

	out = in->partialClone();

	smoothImageCUDA(out, in);

	writeImage(argv[2], out);

	delete in;
	delete out;
	
	return 0;
}
