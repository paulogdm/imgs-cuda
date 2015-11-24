#ifndef IMAGE_H
#define IMAGE_H


#define UNK_TYPE	0
#define GRAY_TYPE 	1	
#define RGB_TYPE	2

#define GRAY_EXT 	".pgm"
#define RGB_EXT 	".ppm"

class Image{
private:
	unsigned char *data;
	int rows, cols;
	char type[3];
	
	void dataFree();

protected:
	void dataAlloc();
	

public:
	Image();
	Image(int rows, int cols);
	virtual ~Image();

	int getRows();
	int getRowsSize(int n_rows);

	int getCols();

	void readFile(const char *name);

	void writeFile(const char *name);

	unsigned char* getData();
	unsigned char* getData(int line_start);
	/*unsigned char* setData();
	unsigned char* setData(int line_start);
*/
	void setType(char *type);
	char* getType();
	
	virtual int getPixelSize();
	virtual Image *partialClone();

};

class grayImage : public Image {
public:
	grayImage();
	grayImage(int rows, int columns);
	~grayImage();
	int getPixelSize();
	virtual grayImage *partialClone();
};

class rgbImage : public Image {
public:
	rgbImage();
	rgbImage(int rows, int columns);
	~rgbImage();
	int getPixelSize();
	virtual rgbImage *partialClone();
};

int getImageType(const char *name);

Image* createImage(int type);
Image* createImage(const char *name);
Image* createImage(int type, int rows, int cols);

#endif
