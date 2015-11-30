#include "image.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

IMAGE_CALL_LEVEL
void Image::dataAlloc(){
	if(this->cols < 1 || this->rows < 1){
		this->data = NULL;
	} else {
		this->data = (unsigned char*) malloc(this->getPixelSize()*(this->cols)*(this->rows));
	}
}

IMAGE_CALL_LEVEL
void Image::dataFree(){
	free(this->data);
}

IMAGE_CALL_LEVEL
Image::Image(){
	Image(0,0);
}

IMAGE_CALL_LEVEL
Image::Image(int rows, int cols){

	this->rows = rows;
	this->cols = cols;
	this->data = NULL;
}

IMAGE_CALL_LEVEL
Image::~Image(){
	if(this->data != NULL)
		free(this->data);
}

IMAGE_CALL_LEVEL
int Image::getRows(){
	return this->rows;
}

IMAGE_CALL_LEVEL
int Image::getRowsSize(int n_rows){
	return this->getPixelSize()*(this->cols)*(n_rows);
}

IMAGE_CALL_LEVEL
int Image::getCols(){
	return this->cols;
}

IMAGE_CALL_LEVEL
void Image::readFile(const char *name){
	FILE *file;
	char type[3] = "";
	char junk;
	int total_size;
	int cols, rows;
	int max_value;
	file = fopen(name, "rb");

	if(file == NULL)
		return;

	fread(type, sizeof(char), 2, file);
	strcpy(this->type, type);

	fread(&junk, sizeof(char), 1, file);

	fread(&junk, sizeof(char), 1, file);

	if(junk == '#'){
		do{
			fread(&junk, sizeof(char), 1, file);
		}while(junk != '\n');
	} else {
		fseek(file, -sizeof(char), SEEK_CUR);
	}

	fscanf(file, "%d %d\n", &cols, &rows);
	fscanf(file, "%d\n", &max_value);

	this->rows = rows;
	this->cols = cols;

	this->dataAlloc();

	total_size = rows*cols*this->getPixelSize();

	fread(this->data, sizeof(unsigned char), total_size,file);
}

IMAGE_CALL_LEVEL
void Image::writeFile(const char *name){
	
	FILE *file;
	int total_size;
	int max_value = 255;

	file = fopen(name, "wb+");

	fprintf(file, "%s\n", this->type);
	fprintf(file, "%d %d\n", this->cols, this->rows);
	fprintf(file, "%d\n", max_value);

	total_size = this->rows*this->cols*this->getPixelSize();

	fwrite(this->data, sizeof(unsigned char), total_size, file);
}

IMAGE_CALL_LEVEL
unsigned char* Image::getData(){
	return this->data;
}

IMAGE_CALL_LEVEL
unsigned char* Image::getData(int line_start){
	if(line_start < this->getRows())
		return(this->data + (this->getPixelSize()*this->getCols())*line_start*sizeof(unsigned char));
	return NULL;
}

IMAGE_CALL_LEVEL
int Image::getPixelSize(){
	return 0;
}

IMAGE_CALL_LEVEL
Image* Image::partialClone(){
	Image* copy = new Image(this->getRows(), this->getRows());
	return copy;
}

IMAGE_CALL_LEVEL
void Image::setType(char *type){
	strcpy(this->type, type);
}

IMAGE_CALL_LEVEL
char* Image::getType(){
	return this->type;
}

////////////////
///GRAY IMAGE //
////////////////
IMAGE_CALL_LEVEL
grayImage::grayImage():
Image(){}

IMAGE_CALL_LEVEL
grayImage::grayImage(int rows, int columns):
Image(rows, columns){
	this->dataAlloc();
}

IMAGE_CALL_LEVEL
grayImage::~grayImage(){

}

IMAGE_CALL_LEVEL
int grayImage::getPixelSize(){
	return sizeof(unsigned char);
}

IMAGE_CALL_LEVEL
grayImage* grayImage::partialClone(){
	grayImage* copy = new grayImage(this->getRows(), this->getCols());

	copy->setType(this->getType());
	
	return copy;
}

///////////////
///RGB IMAGE //
///////////////
IMAGE_CALL_LEVEL
rgbImage::rgbImage():
Image(){}
	
IMAGE_CALL_LEVEL
rgbImage::rgbImage(int rows, int columns):
Image(rows, columns){
	this->dataAlloc();
}

IMAGE_CALL_LEVEL
rgbImage::~rgbImage(){

}
	
IMAGE_CALL_LEVEL
int rgbImage::getPixelSize(){
	return 3*sizeof(unsigned char);
}

IMAGE_CALL_LEVEL
rgbImage* rgbImage::partialClone(){	
	rgbImage* copy = new rgbImage(this->getRows(), this->getCols());
	
	copy->setType(this->getType());

	return copy;
}

///////////////////////
///SUPPORT FUNCTIONS //
///////////////////////
IMAGE_CALL_LEVEL
int getImageType(const char *name){

	if(memcmp(GRAY_EXT, name + strlen(name)-4, 4*sizeof(char)) == 0){
		return GRAY_TYPE;
	} if(memcmp(RGB_EXT, name + strlen(name)-4, 4*sizeof(char)) == 0){
		return RGB_TYPE;
	}

	return UNK_TYPE;
}

IMAGE_CALL_LEVEL
Image* createImage(int type){
	Image *buffer = NULL;
	
	if(type == GRAY_TYPE){	
		buffer = new grayImage();
	} else if(type == RGB_TYPE){
		buffer = new rgbImage();
	}

	return buffer;
}

IMAGE_CALL_LEVEL
Image* createImage(int type, int rows, int cols){
	Image *buffer = NULL;
	
	if(type == GRAY_TYPE){	
		buffer = new grayImage(rows, cols);
	} else if(type == RGB_TYPE){
		buffer = new rgbImage(rows, cols);
	}

	return buffer;
}

IMAGE_CALL_LEVEL
Image* createImage(const char *name){
	return createImage(getImageType(name));
}

