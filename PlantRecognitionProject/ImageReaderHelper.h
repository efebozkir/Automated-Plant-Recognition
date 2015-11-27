#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<cstdlib>

#include "opencv\cv.h"

using namespace std;
using namespace cv;

class ImageReaderHelper
{
	IplImage* readImage;


	public:
	void setReadImage(IplImage* read);
	IplImage* getReadImage();
	IplImage* readBinaryImage();

};