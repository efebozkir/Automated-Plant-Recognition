#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<cstdlib>
#include <fstream>
#include <windows.h>
#include "dirent.h"
#include <string>
#include <cmath>

#include "opencv\cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv\highgui.h"
#include "Segment.h"


void Segment::makeSegmentation()
{
	Mat image;
    image= cv::imread("C:/Acer campestre-1153.jpg");
	inputImage=cvCloneImage(&(IplImage)image);

    // define bounding rectangle 
    cv::Rect rectangle(80,70,image.cols-150,image.rows-100);

    cv::Mat result; // segmentation result (4 possible values)
    cv::Mat bgModel,fgModel; // the models (internally used)

    // GrabCut segmentation
    cv::grabCut(image,    // input image
                    result,   // segmentation result
                            rectangle,// rectangle containing foreground 
                            bgModel,fgModel, // models
                            1,        // number of iterations
                            cv::GC_INIT_WITH_RECT); // use rectangle
    cout << "GrabCut Done." <<endl;
    // Get the pixels marked as likely foreground
    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
    // Generate output image
    cv::Mat foreground(image.size(),CV_8UC3,cv::Scalar(255,255,255));
    image.copyTo(foreground,result); // bg pixels not copied

    // draw rectangle on original image
    cv::rectangle(image, rectangle, cv::Scalar(255,255,255),1);
    cv::namedWindow("Image");
    cv::imshow("Image",image);

    // display result
    //cv::namedWindow("Segmented Image");
    //cv::imshow("Segmented Image",foreground);

	//Mat image1;
	IplImage* image2=cvCloneImage(&(IplImage)foreground);
	//cvShowImage("temp", image2);
	IplImage* g_gray = cvCreateImage( cvGetSize( image2 ), 8, 1 ); // grayscale hali 

	cvCvtColor( image2, g_gray, CV_BGR2GRAY );
	cvThreshold( g_gray, g_gray, 175, 255, CV_THRESH_BINARY );	// orjinal binary hali */
	//cvShowImage("tempgray", g_gray);
	int height    = g_gray->height;
    int width     = g_gray->width;
    int step      = g_gray->widthStep;
    int channels  = g_gray->nChannels;
    uchar * data      = (uchar *)g_gray->imageData;
	 // invert the image
	for(int i=0;i<height;i++)
     for(int j=0;j<width;j++)
        for(int k=0;k<channels;k++)  //loop to read for each channel
           data[i*step+j*channels+k]=255-data[i*step+j*channels+k];    //inverting the image
 
	// show the image
	//cvShowImage("example2", g_gray );//g_gray sondaki segmente edilmiþ image
	segmentedImage=cvCloneImage(g_gray);
	//cvShowImage("segmentedImage", segmentedImage );

};

void Segment::setInputImage(IplImage* setInputImage)
{
	inputImage=setInputImage;
};

IplImage* Segment::getInputImage()
{
	return inputImage;
};

void Segment::setSegmentedImage(IplImage* setSegmentedImage)
{
	segmentedImage=setSegmentedImage;
};

IplImage* Segment::getSegmentedImage()
{
	return segmentedImage;
};