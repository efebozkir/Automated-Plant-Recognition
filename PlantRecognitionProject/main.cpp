#define _CRT_SECURE_NO_DEPRECATE
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv\highgui.h"
#include <ml.h>	
#include "opencv\cv.h"
#include "opencv2\core\core.hpp"

//#include "opencv2\nonfree\features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\nonfree\nonfree.hpp"
#include "opencv2\features2d\features2d.hpp"


//#include "ExtractDescriptorHelper.h"


#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <sys/types.h>
#include <fstream>


#include "ImageReaderHelper.h"
#include "dirent.h"
#include "Classify.h"
#include "ClassifyIndividual.h"
#include <string>
#include <windows.h>
#include "NormaliseGeoFeatures.h"
#include "Segment.h"
#include "ConvertToCsv.h"
#include "isLeaf.h"


using namespace cv;
using namespace std;

#define NUMBER_OF_TRAINING_SAMPLES 1050 //1050 for swedish leaves, 2753 for imageclef
#define ATTRIBUTES_PER_SAMPLE 159  // 150 fourier + 9 geometric
#define NUMBER_OF_TESTING_SAMPLES 75  // 75 for swedish leaves, 48 imageclef

#define NUMBER_OF_CLASSES 15 // 15 for swedish leaves, 69 for imageclef

// loads the sample database from file (which is a CSV text file)

int read_data_from_csv(const char* filename, Mat data, Mat classes,
                       int n_samples )
{
    float tmp;
	int counter=0;

    // if we can't read the input file then return 0
    FILE* f = fopen( filename, "r" );
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
    }

    // for each sample in the file

    for(int line = 0; line < n_samples; line++)
    {

        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++)
        {
            if (attribute < 159)
            {

                // first 64 elements (0-63) in each line are the attributes

                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;
                //printf("%f,", data.at<float>(line, attribute));
				

            }
            else if (attribute == 159)
            {
				
                // attribute 65 is the class label {0 ... 9}

                fscanf(f, "%f,", &tmp);
                classes.at<float>(line, 0) = tmp;
				//printf("%f Counter: %d\n", &tmp, counter);
				//counter++;
				
            }
        }
    }
	cout<<"Counter "<<counter<<endl;
    fclose(f);

    return 1; // all OK
}


/** @function main */
int main( int argc, char** argv )
{
	

	
	char selection;
	cout<<"Welcome to Plant Recognition System"<<endl;
	cout<<"Please select following in order to make an operation:"<<endl;
	cout<<"S for Segmentation and Feature Extraction of Normal Leaf Image"<<endl;
	cout<<"F for Feature Extraction of a Binary Image"<<endl;
	cout<<"C for Classification of Test Set with NN"<<endl;
	cout<<"T for Feature Extraction and Training of Train Set"<<endl;
	cout<<"Q for Extracted Features to CSV File"<<endl;
	cout<<"R for Classification with Random Forests"<<endl;
	cout<<"E for SIFT "<<endl;
	cout<<"Y for leaf detection test with SIFT+BoF+SVM"<<endl;
	cin>>selection;


	switch(selection)
	{

		case 'e':
		case 'E':
			{
				
				/*
				IplImage* input=cvLoadImage("C:/fb2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
				vector<KeyPoint> keypoints;
				OutputArray descriptors;
				InputArray mask;*/

				

				
				
				
				
				
		
				/*
				Mat input=imread("C:/fb2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
				
				if( !input.data  )
				{
					cout<<"Error while loading data"<<endl;
					return -1;
				}
				
				int minHessian = 400;
				SurfFeatureDetector detector( minHessian );
				
				std::vector<KeyPoint> keypoints_1;
				detector.detect( input, keypoints_1 );
				
				Mat img_keypoints_1; 
				drawKeypoints( input, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
				imshow("Keypoints 1", img_keypoints_1 );*/
				
				/*
				const cv::Mat input = cv::imread("C:/MyPic.png", 0); //Load as grayscale

				cv::SiftFeatureDetector detector;
				std::vector<cv::KeyPoint> keypoints;
				detector.detect(input, keypoints);
				std::vector<cv::Point2f> points;
				std::vector<cv::KeyPoint>::iterator it;
				*/






				//cv::Mat pointMatrix(points);

		

				
				// Add results to image and save.
				/*cv::Mat output;
				cv::drawKeypoints(input, keypoints, output);
				cv::imwrite("C:/sift_result.jpg", output);*/

				//waitKey(0);
				//return 0;


			}
			break;

	////TRAIN VE TEST FEATURELARI CSV DOSYASI OLARAK TUTULDU (RANDOM FOREST ICIN)
		case 'q':
		case 'Q':
			{
				ConvertToCSV* myCsv=new ConvertToCSV();
				myCsv->TrainFeaturesAsCSV();
				myCsv->TestFeaturesAsCSV();
			}
		break;


	////SEGMENTATION WITH GRABCUT THEN FEATURE EXTRACTION
		case 'S':
		case 's':
			{
				Segment *mySegment=new Segment();
				mySegment->makeSegmentation();

				IplImage* segmented=cvCloneImage(mySegment->getSegmentedImage());
				//cvShowImage("segmentedImage", segmented );

				IplImage* converted=cvCreateImage( cvGetSize( segmented ), 8, 3 );
				cvCvtColor(segmented, converted, CV_GRAY2RGB);
				cvShowImage("converted", converted );
				ExtractDescriptorHelper* myExtract=new ExtractDescriptorHelper();
				myExtract->ExtractDescriptors(converted);
			}
		break;
		
	////FEATURE EXTRACTION OF SINGLE IMAGE
		case 'F':
		case 'f':
			{
				ImageReaderHelper* tmpReader=new ImageReaderHelper();
				IplImage* src= tmpReader->readBinaryImage();
				ExtractDescriptorHelper* tmpDescriptorFinder=new ExtractDescriptorHelper();
				tmpDescriptorFinder->ExtractDescriptors(src);
				tmpDescriptorFinder->sortVector();
				tmpDescriptorFinder->createFeatureVector();
			}
		break;

	////GEOMETRIC FEATURELARIN NORMALIZASYONU
	/*
	NormaliseGeoFeatures *normaliseTmp=new NormaliseGeoFeatures();
	//normaliseTmp->produceFileNamesVect();
	normaliseTmp->initializeMaxMin();
	normaliseTmp->calcMaxMin();
	normaliseTmp->normalizeGeoFeatures();
	normaliseTmp->writeMaxMinToFile();
	*/
	
	////CLASSIFICATION WITH NEAREST NEIGHBOUR
		case 'C':
		case 'c':
			{
				int b;
				Classify *temp=new Classify();
				//temp->getMinMaxFromFile();
				b=temp->makeClassification();
				//temp->sortDataVect();
				//temp->printVector();
			}
		break;
		
	/*
	int d;
	ClassifyIndividual *myTemp=new ClassifyIndividual();
	myTemp->makeClassification();
	myTemp->sortDataVect();
	myTemp->printVector();
	*/
	
	
	//FEATURE EXTRACTION AND WRITING IT TO TXT
	
	//DOSYADAN OKUMA VE FEATURELARI TXT DOSYASINA YAZMA YAPILDI, ÇALIŞIYOR.
	//C:/Deneme/ DİZİNİNDEKİ DOSYALAR İÇİN GERÇEKLENDİ, FEATURELAR YAZILIYOR..
		case 'T':
		case 't':
			{
				int sayac=0;
				DIR *dir;
				struct dirent *ent;
				char folder[100];
				char writeFile[100];
				string write;
				string str1="C:/TrainSet/.";
				string str2="C:/TrainSet/..";
				string str3="C:/TrainSet/Thumbs.db";
				if ((dir = opendir ("C:/TrainSet/"))) 
					{
			
						while ((ent = readdir (dir)) != NULL) 
						{
								sprintf (folder, "C:/TrainSet/%s", ent->d_name);
								sprintf (writeFile, "C:/Features/%s.txt", ent->d_name);
								cout<<"File name:"<<folder<<endl;
					
								if(str1.compare(folder)==0 || str2.compare(folder)==0 || str3.compare(folder)==0)
									continue;
					
								IplImage* src;
								src=cvLoadImage(folder);
								ExtractDescriptorHelper* tmpDescriptorFinder=new ExtractDescriptorHelper();
								tmpDescriptorFinder->ExtractDescriptors(src);
					
								tmpDescriptorFinder->sortVector();
								tmpDescriptorFinder->createFeatureVector();
								vector<double> writeVector=tmpDescriptorFinder->getMyFeatureVector();
								double featureArray[featureVectSize];
					
								free(tmpDescriptorFinder);
								sayac++;

								for(int p=0; p<featureVectSize; p++)
									featureArray[p]=writeVector.at(p);

								ofstream myfile;
								myfile.open (writeFile);
								for(int k=0; k<featureVectSize; k++)
									myfile << featureArray[k]<<"\n";
								myfile.close();
					
					
						}
					closedir (dir);
					} 
		
					else {
			
						cout<<"Error exists"<<endl;
						perror ("");
						return EXIT_FAILURE;
						}

					cout<<"Sayac: "<<sayac<<endl;
			
			}
		break;

		case 'R':
		case 'r':
			{
				// lets just check the version first
	
				printf ("OpenCV version %s (%d.%d.%d)\n", CV_VERSION,
							CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);

				// define training data storage matrices (one for attribute examples, one
				// for classifications)
				//CV_8UC(15) , CV_32FC1
				Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
				Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

				//define testing data storage matrices

				Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
				Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);

				// define all the attributes as numerical
				// alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
				// that can be assigned on a per attribute basis

				Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 1, 1, CV_8U );
				var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

				// this is a classification problem (i.e. predict a discrete number of class
				// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

				var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;

				double result; // value returned from a prediction

				// load training and testing data sets

				if (read_data_from_csv(argv[1], training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES) &&
						read_data_from_csv(argv[2], testing_data, testing_classifications, NUMBER_OF_TESTING_SAMPLES))
				{
					// define the parameters for training the random forest (trees)

					float priors[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};  // weights of each classification for classes
					// (all equal as equal samples of each digit)

					CvRTParams params = CvRTParams(20, // max depth
												5, // min sample count
												0, // regression accuracy: N/A here
												false, // compute surrogate split, no missing data
												15, // max number of categories (use sub-optimal algorithm for larger numbers)
												priors, // the array of priors
												false,  // calculate variable importance
												40,       // number of variables randomly selected at node and used to find the best split(s).
												100,	 // max number of trees in the forest
												0.01f,				// forrest accuracy
												CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination cirteria
												);

					// train random forest classifier (using training data)

					printf( "\nUsing training database: %s\n\n", argv[1]);
					CvRTrees* rtree = new CvRTrees;

					rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
								 Mat(), Mat(), var_type, Mat(), params);

					// perform classifier testing and report results

					Mat test_sample;
					int correct_class = 0;
					int wrong_class = 0;
					int false_positives [NUMBER_OF_CLASSES] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

					printf( "\nUsing testing database: %s\n\n", argv[2]);

					for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
					{

						// extract a row from the testing matrix

						test_sample = testing_data.row(tsample);

						// run random forest prediction

						result = rtree->predict(test_sample, Mat());

						printf("Testing Sample %i -> class result (digit %d)\n", tsample, (int) result);

						// if the prediction and the (true) testing classification are the same
						// (N.B. openCV uses a floating point decision tree implementation!)

						if (fabs(result - testing_classifications.at<float>(tsample, 0))
								>= FLT_EPSILON)
						{
							// if they differ more than floating point error => wrong class

							wrong_class++;

							false_positives[(int) result]++;

						}
						else
						{

							// otherwise correct

							correct_class++;
						}
					}

					printf( "\nResults on the testing database: %s\n"
							"\tCorrect classification: %d (%g%%)\n"
							"\tWrong classifications: %d (%g%%)\n",
							argv[2],
							correct_class, (double) correct_class*100/NUMBER_OF_TESTING_SAMPLES,
							wrong_class, (double) wrong_class*100/NUMBER_OF_TESTING_SAMPLES);

					for (int i = 0; i < NUMBER_OF_CLASSES; i++)
					{
						printf( "\tClass (digit %d) false postives 	%d (%g%%)\n", i,
								false_positives[i],
								(double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
					}


					// all matrix memory free by destructors


					// all OK : main returns 0

					return 0;
				}

				// not OK : main returns -1

				return -1;
			
			}
		break;

		case 'Y':
		case 'y':
			{
				isLeaf *myLeafTmp=new isLeaf();

				//myLeafTmp->produceDictionary();  *** herhangi 2sini veya 3unu aynı anda çalıştırma.
				myLeafTmp->produceTrainData();  // everything here.
				//myLeafTmp->produceTestData();
				//myLeafTmp->isLeafOrNot();  crashes
			}
			break;
	}
		
	waitKey(0);
	return 0;
};


