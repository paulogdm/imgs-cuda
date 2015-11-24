#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <image.h>
#include <smooth.h>

#define UNK_TYPE	0
#define GRAY_TYPE 	1	
#define RGB_TYPE	2

#define GRAY_EXT 	".pgm"
#define RGB_EXT 	".ppm"


Image* readImage(const char *name){

	Image *buffer = NULL;
	
	if(memcmp(GRAY_EXT, name + strlen(name)-4, 4*sizeof(char)) == 0){
		buffer = new grayImage();
	} if(memcmp(RGB_EXT, name + strlen(name)-4, 4*sizeof(char)) == 0){
		buffer = new rgbImage();
	}

	if(buffer != NULL)
		buffer->readFile(name);

	return buffer;
}

void writeImage(const char *name, Image *out){
	if(out != NULL)
		out->writeFile(name);
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
	
	smoothImage(out, in);

	writeImage(argv[2], out);

	delete in;
	delete out;
	
	return 0;
}
