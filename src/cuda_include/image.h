#ifndef IMAGE_H
#define IMAGE_H

#define UNK_TYPE	0
#define GRAY_TYPE 	1	
#define RGB_TYPE	2

#define GRAY_EXT 	".pgm"
#define RGB_EXT 	".ppm"

#ifdef __CUDACC__
#define IMAGE_CALL_LEVEL __host__
#else
#define SMOOTH_CALL_LEVEL
#endif


class Image{
private:
	unsigned char *data;
	char type[3];
	

protected:
	IMAGE_CALL_LEVEL
	void dataAlloc();
	
	IMAGE_CALL_LEVEL
	void dataFree();
	

public:
	int rows, cols;
	
	IMAGE_CALL_LEVEL
	Image();

	IMAGE_CALL_LEVEL
	Image(int rows, int cols);

	IMAGE_CALL_LEVEL
	virtual ~Image();

	IMAGE_CALL_LEVEL
	int getRows();
	IMAGE_CALL_LEVEL
	int getRowsSize(int n_rows);

	IMAGE_CALL_LEVEL
	int getCols();

	IMAGE_CALL_LEVEL
	void readFile(const char *name);

	IMAGE_CALL_LEVEL
	void writeFile(const char *name);

	IMAGE_CALL_LEVEL
	unsigned char* getData();

	IMAGE_CALL_LEVEL
	unsigned char* getData(int line_start);

	IMAGE_CALL_LEVEL
	void setType(char *type);

	IMAGE_CALL_LEVEL
	char* getType();
	
	IMAGE_CALL_LEVEL
	virtual int getPixelSize();

	IMAGE_CALL_LEVEL
	virtual Image *partialClone();

};

class grayImage : public Image {
public:
	
	grayImage();
	
	grayImage(int rows, int columns);

	~grayImage();
	
	IMAGE_CALL_LEVEL
	int getPixelSize();
	

	virtual grayImage *partialClone();
};

class rgbImage : public Image {
public:
	
	
	rgbImage();
	
	
	rgbImage(int rows, int columns);
	

	~rgbImage();
	
	IMAGE_CALL_LEVEL
	int getPixelSize();

	IMAGE_CALL_LEVEL
	virtual rgbImage *partialClone();
};

IMAGE_CALL_LEVEL
int getImageType(const char *name);

IMAGE_CALL_LEVEL
Image* createImage(int type);

IMAGE_CALL_LEVEL
Image* createImage(const char *name);

IMAGE_CALL_LEVEL
Image* createImage(int type, int rows, int cols);

#endif
