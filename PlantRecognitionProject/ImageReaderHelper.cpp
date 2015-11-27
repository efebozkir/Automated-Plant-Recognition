#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<cstdlib>

#include "opencv\cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv\highgui.h"

#include "ImageReaderHelper.h"

using namespace std;
using namespace cv;


IplImage* ImageReaderHelper::getReadImage()
{
	return this->readImage;
};

void ImageReaderHelper::setReadImage(IplImage* setImage)
{
	this->readImage=setImage;
};

IplImage* ImageReaderHelper::readBinaryImage()
{
	
	IplImage* src;
	char *filename="C:/MyPic4.png";
	src=cvLoadImage(filename);

	return src;
	
};

