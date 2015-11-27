#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include "opencv\cv.h"
#include <cstring>

using namespace std;
using namespace cv;

class Segment
{
	protected:
	IplImage* inputImage;
	IplImage* segmentedImage;

	public:
	
	void makeSegmentation();
	void setInputImage(IplImage*);
	IplImage* getInputImage();
	void setSegmentedImage(IplImage*);
	IplImage* getSegmentedImage();

};