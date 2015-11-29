#ifndef IMAGE_H
#define IMAGE_H

#ifdef __CUDACC__
#define CUDA_CALL	__host__ __device__
#else
#define CUDA_CALL
#endif 

#define UNK_TYPE	0
#define GRAY_TYPE 	1	
#define RGB_TYPE	2

#define GRAY_EXT 	".pgm"
#define RGB_EXT 	".ppm"

class Image{
private:
	unsigned char *data;
	char type[3];
	

protected:
	__host__
	void dataAlloc();
	
	__host__
	void dataFree();
	

public:
	int rows, cols;
	
	__host__
	Image();

	__host__
	Image(int rows, int cols);

	__host__
	virtual ~Image();

	CUDA_CALL
	int getRows();
	CUDA_CALL
	int getRowsSize(int n_rows);

	CUDA_CALL
	int getCols();

	__host__
	void readFile(const char *name);

	__host__
	void writeFile(const char *name);

	CUDA_CALL
	unsigned char* getData();

	CUDA_CALL
	unsigned char* getData(int line_start);

	__host__
	void setType(char *type);

	__host__
	char* getType();
	
	CUDA_CALL
	virtual int getPixelSize();

	__host__
	virtual Image *partialClone();

};

class grayImage : public Image {
public:
	
	grayImage();
	
	grayImage(int rows, int columns);

	~grayImage();
	
	CUDA_CALL
	int getPixelSize();
	

	virtual grayImage *partialClone();
};

class rgbImage : public Image {
public:
	
	
	rgbImage();
	
	
	rgbImage(int rows, int columns);
	

	~rgbImage();
	
	CUDA_CALL
	int getPixelSize();

	__host__
	virtual rgbImage *partialClone();
};

__host__
int getImageType(const char *name);

__host__
Image* createImage(int type);

__host__
Image* createImage(const char *name);

__host__
Image* createImage(int type, int rows, int cols);

#endif
