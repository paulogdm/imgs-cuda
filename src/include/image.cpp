#include "image.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

void Image::dataAlloc(){
	if(this->cols < 1 || this->rows < 1){
		this->data = NULL;
	} else {
		this->data = (unsigned char*) malloc(this->getPixelSize()*(this->cols)*(this->rows));
	}
}

void Image::dataFree(){
	free(this->data);
}

Image::Image(){
	Image(0,0);
}

Image::Image(int rows, int cols){

	this->rows = rows;
	this->cols = cols;
	this->data = NULL;
}

Image::~Image(){
	if(this->data != NULL)
		free(this->data);
}

int Image::getRows(){
	return this->rows;
}

int Image::getRowsSize(int n_rows){
	return this->getPixelSize()*(this->cols)*(n_rows);
}

int Image::getCols(){
	return this->cols;
}

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

unsigned char* Image::getData(){
	return this->data;
}

unsigned char* Image::getData(int line_start){
	if(line_start < this->getRows())
		return(this->data + (this->getPixelSize()*this->getCols())*line_start*sizeof(unsigned char));
	return NULL;
}

int Image::getPixelSize(){
	return 0;
}

Image* Image::partialClone(){
	Image* copy = new Image(this->getRows(), this->getRows());
	return copy;
}

void Image::setType(char *type){
	strcpy(this->type, type);
}

char* Image::getType(){
	return this->type;
}

////////////////
///GRAY IMAGE //
////////////////
grayImage::grayImage():
Image(){}

grayImage::grayImage(int rows, int columns):
Image(rows, columns){
	this->dataAlloc();
}

grayImage::~grayImage(){

}

int grayImage::getPixelSize(){
	return sizeof(unsigned char);
}

grayImage* grayImage::partialClone(){
	grayImage* copy = new grayImage(this->getRows(), this->getCols());

	copy->setType(this->getType());
	
	return copy;
}

///////////////
///RGB IMAGE //
///////////////
rgbImage::rgbImage():
Image(){}
	
rgbImage::rgbImage(int rows, int columns):
Image(rows, columns){
	this->dataAlloc();
}

rgbImage::~rgbImage(){

}
	
int rgbImage::getPixelSize(){
	return 3*sizeof(unsigned char);
}

rgbImage* rgbImage::partialClone(){	
	rgbImage* copy = new rgbImage(this->getRows(), this->getCols());
	
	copy->setType(this->getType());

	return copy;
}

///////////////////////
///SUPPORT FUNCTIONS //
///////////////////////
int getImageType(const char *name){

	if(memcmp(GRAY_EXT, name + strlen(name)-4, 4*sizeof(char)) == 0){
		return GRAY_TYPE;
	} if(memcmp(RGB_EXT, name + strlen(name)-4, 4*sizeof(char)) == 0){
		return RGB_TYPE;
	}

	return UNK_TYPE;
}

Image* createImage(int type){
	Image *buffer = NULL;
	
	if(type == GRAY_TYPE){	
		buffer = new grayImage();
	} else if(type == RGB_TYPE){
		buffer = new rgbImage();
	}

	return buffer;
}

Image* createImage(int type, int rows, int cols){
	Image *buffer = NULL;
	
	if(type == GRAY_TYPE){	
		buffer = new grayImage(rows, cols);
	} else if(type == RGB_TYPE){
		buffer = new rgbImage(rows, cols);
	}

	return buffer;
}

Image* createImage(const char *name){
	return createImage(getImageType(name));
}

