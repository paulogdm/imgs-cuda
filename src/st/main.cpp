#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <image.h>
#include <smooth.h>
#include <time.h>

#define WRITE_IMAGE_OUT		false

#define EXEC_N_TIMES		10



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

int main(int argc, const char **argv){

	Image *in;
	Image *out;
	clock_t start, end;

	if(argc != 3){
		printf("Usage: %s <IMAGE_IN> <IMAGE_OUT>\n", argv[0]);
		return 1;
	}

	in = readImage(argv[1]);

	if(in->getData() == NULL){
		printf("File does not seem to exist\n");
		return 1;
	}

	out = in->partialClone();
	for(int n_exec = 0; n_exec < EXEC_N_TIMES; n_exec++){

	start = clock();
	smoothImage(out, in);
	end = clock();

		printf("Image: %s\n", argv[1]);
		printf("Smooth time: %.4lf\n", ((double) (end - start))/CLOCKS_PER_SEC);
		printf("\n");
	}

	if(WRITE_IMAGE_OUT)
		writeImage(argv[2], out);

	delete in;
	delete out;
	
	return 0;
}
