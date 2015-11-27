#define _CRT_SECURE_NO_DEPRECATE
#include<iostream>
#include<cstdlib>
#include <fstream>
#include <windows.h>

#include <string>
#include <cmath>


#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2\features2d\features2d.hpp>

#include "isLeaf.h"
#include "ExtractDescriptorHelper.h"


#define Y_NUMBER_OF_TRAINING_SAMPLES 1344
#define Y_NUMBER_OF_TEST_SAMPLES 344
#define NUMBER_OF_ATTRIBUTES 159
 
void isLeaf::featureExtractSamples()
{

	DIR *dirTest;
	struct dirent *entTest;
	string str3="C:/Y_TestSet/.";
	string str4="C:/Y_TestSet/..";
	string str5="C:/Y_TestSet/Thumbs.db";
	char FileName[100];
	char writeFile[100];

	if ((dirTest = opendir ("C:/Y_TestSet/"))) 
	{
		while ((entTest = readdir (dirTest)) != NULL) 
		{
					sprintf (FileName, "C:/Y_TestSet/%s", entTest->d_name);
					sprintf (writeFile, "C:/Y_TestSetFeatures/%s.txt", entTest->d_name);
					if(str3.compare(FileName)==0 || str4.compare(FileName)==0 || str5.compare(FileName)==0)
						continue;
					

					cout<<"File name: "<<FileName<<endl;
					
	IplImage* src;
	src=cvLoadImage(FileName);
	ExtractDescriptorHelper* tmpDescriptorFinder=new ExtractDescriptorHelper();
	tmpDescriptorFinder->ExtractDescriptors(src);
					
	tmpDescriptorFinder->sortVector();
	tmpDescriptorFinder->createFeatureVector();

					ofstream myfile;
					myfile.open (writeFile);
					for(int k=0; k<NUMBER_OF_ATTRIBUTES; k++)
						myfile << tmpDescriptorFinder->getMyFeatureVector().at(k)<<"\n";
					myfile.close();

		} 
		closedir (dirTest);
	} 
	else 
	{
		cout<<"Error exists"<<endl;
		perror ("");
	}
	
	
	
	
	DIR *dirTrain;
	struct dirent *entTrain;
	string str3Train="C:/Y_TrainSet/.";
	string str4Train="C:/Y_TrainSet/..";
	string str5Train="C:/Y_TrainSet/Thumbs.db";
	char FileNameTrain[100];
	char writeFileTrain[100];

	if ((dirTrain = opendir ("C:/Y_TrainSet/"))) 
	{
		while ((entTrain = readdir (dirTrain)) != NULL) 
		{
					sprintf (FileNameTrain, "C:/Y_TrainSet/%s", entTrain->d_name);
					sprintf (writeFileTrain, "C:/Y_TrainSetFeatures/%s.txt", entTrain->d_name);
					if(str3Train.compare(FileNameTrain)==0 || str4Train.compare(FileNameTrain)==0 || str5Train.compare(FileNameTrain)==0)
						continue;
					

					cout<<"File name: "<<FileNameTrain<<endl;
					
	IplImage* srcTrain;
	srcTrain=cvLoadImage(FileNameTrain);
	ExtractDescriptorHelper* tmpDescriptorFinderTrain=new ExtractDescriptorHelper();
	tmpDescriptorFinderTrain->ExtractDescriptors(srcTrain);
					
	tmpDescriptorFinderTrain->sortVector();
	tmpDescriptorFinderTrain->createFeatureVector();

					ofstream myfileTrain;
					myfileTrain.open (writeFileTrain);
					for(int k=0; k<NUMBER_OF_ATTRIBUTES; k++)
						myfileTrain << tmpDescriptorFinderTrain->getMyFeatureVector().at(k)<<"\n";
					myfileTrain.close();

		} 
		closedir (dirTrain);
	} 	
	else 
	{
		cout<<"Error exists"<<endl;
		perror ("");
	}
	
};

void isLeaf::csvOfSamples()
{
	/*
	DIR *dirTrain;
	struct dirent *entTrain;
	char folderTrain[100];
	string str1Train="C:/Y_TrainSetFeatures/.";
	string str2Train="C:/Y_TrainSetFeatures/..";
	double readArrayTrain[NUMBER_OF_ATTRIBUTES];

	if ((dirTrain = opendir ("C:/Y_TrainSetFeatures/"))) 
		{
			while ((entTrain = readdir (dirTrain)) != NULL) 
			{

					sprintf (folderTrain, "C:/Y_TrainSetFeatures/%s", entTrain->d_name);
					//cout<<"File name:"<<folder<<endl;
					
					if(str1Train.compare(folderTrain)==0 || str2Train.compare(folderTrain)==0)
						continue;
					
					ifstream myfileTrain;
					
					myfileTrain.open(folderTrain);
						for(int i=0; i<NUMBER_OF_ATTRIBUTES; i++)
						{
							myfileTrain>>readArrayTrain[i];
						}

					string cutTheNameTrain=entTrain->d_name;
					string splitterTrain="-";
					size_t foundTrain=cutTheNameTrain.find(splitterTrain);
					string trainTypeNameTrain=cutTheNameTrain.substr(0,foundTrain); //Train resminin tam tip adý
						
					ofstream myCsvFileTrain;
					myCsvFileTrain.open("C:/Y_CSV-Features/trainSet.csv", ios::out | ios::app);
					

					for(int j=0; j<NUMBER_OF_ATTRIBUTES; j++)
					{
						myCsvFileTrain<<readArrayTrain[j]<<",";
					}
					myCsvFileTrain<<trainTypeNameTrain;
					myCsvFileTrain<<"\n";
					myCsvFileTrain.close();


			}
		closedir (dirTrain);
		
		} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
		 }


	
	DIR *dirTest;
	struct dirent *entTest;
	char folderTest[100];
	string str1Test="C:/Y_TestSetFeatures/.";
	string str2Test="C:/Y_TestSetFeatures/..";
	double readArrayTest[NUMBER_OF_ATTRIBUTES];

	if ((dirTest = opendir ("C:/Y_TestSetFeatures/"))) 
		{
			while ((entTest = readdir (dirTest)) != NULL) 
			{

					sprintf (folderTest, "C:/Y_TestSetFeatures/%s", entTest->d_name);
					//cout<<"File name:"<<folder<<endl;
					
					if(str1Test.compare(folderTest)==0 || str2Test.compare(folderTest)==0)
						continue;
					
					ifstream myfileTest;
					
					myfileTest.open(folderTest);
						for(int i=0; i<NUMBER_OF_ATTRIBUTES; i++)
						{
							myfileTest>>readArrayTest[i];
						}

					string cutTheNameTest=entTest->d_name;
					string splitterTest="-";
					size_t foundTest=cutTheNameTest.find(splitterTest);
					string trainTypeNameTest=cutTheNameTest.substr(0,foundTest); //Train resminin tam tip adý
						
					ofstream myCsvFileTest;
					myCsvFileTest.open("C:/Y_CSV-Features/testSet.csv", ios::out | ios::app);
					

					for(int j=0; j<NUMBER_OF_ATTRIBUTES; j++)
					{
						myCsvFileTest<<readArrayTest[j]<<",";
					}
					myCsvFileTest<<trainTypeNameTest;
					myCsvFileTest<<"\n";
					myCsvFileTest.close();


			}
		closedir (dirTest);
		
		} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
		 }
	*/

};

int isLeaf::readProduceFromCsv()
{
	/*
	char trainFileName[100]="C:/Y_CSV-Features/trainSet.csv";
	char testFileName[100]="C:/Y_CSV-Features/testSet.csv";
	
	float tmp;
	float tmpTest;
	int counter=0;
	// if we can't read the input file then return 0
    FILE* f = fopen( trainFileName, "r" );
	FILE *fTest=fopen(testFileName, "r");

	myTrainData= Mat(Y_NUMBER_OF_TRAINING_SAMPLES, NUMBER_OF_ATTRIBUTES, CV_32FC1);
	myTestData=Mat(Y_NUMBER_OF_TEST_SAMPLES, NUMBER_OF_ATTRIBUTES, CV_32FC1);

	myTrainDataLabel=Mat(Y_NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);
	
    if( !f )
    {
        printf("ERROR: cannot read file %s\n",  trainFileName);
        return 0; // all not OK
    }

	if( !fTest )
    {
        printf("ERROR: cannot read file %s\n",  testFileName);
        return 0; // all not OK
    }

    // for each sample in the file

	for(int line = 0; line < Y_NUMBER_OF_TRAINING_SAMPLES; line++)
    {
		
        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (NUMBER_OF_ATTRIBUTES + 1); attribute++)
        {
            if (attribute < 159)
            {
                // first 64 elements (0-63) in each line are the attributes

                fscanf(f, "%f,", &tmp);
                myTrainData.at<float>(line, attribute) = tmp;
                //printf("%f,", myTrainData.at<float>(line, attribute));
				
            }
            else if (attribute == 159)
            {
                // attribute 65 is the class label {0 ... 9}

                fscanf(f, "%f,", &tmp);
                myTrainDataLabel.at<float>(line, 0) = tmp;
				//printf("%f Counter: %d\n", &tmp, counter);
				//counter++;
				
            }
        }
    }
	//cout<<"Counter "<<counter<<endl;
    fclose(f);

	for(int line = 0; line < Y_NUMBER_OF_TEST_SAMPLES; line++)
    {
		
        // for each attribute on the line in the file

        for(int attribute = 0; attribute < (NUMBER_OF_ATTRIBUTES + 1); attribute++)
        {
            if (attribute < 159)
            {
                // first 64 elements (0-63) in each line are the attributes

                fscanf(f, "%f,", &tmpTest);
                myTestData.at<float>(line, attribute) = tmpTest;
                //printf("%f,", trainData.at<float>(line, attribute));
				
            }
			/*
            else if (attribute == 159)
            {
                // attribute 65 is the class label {0 ... 9}

                fscanf(f, "%f,", &tmp);
                myTrainDataLabel.at<float>(line, 0) = tmp;
				printf("%f Counter: %d\n", &tmp, counter);
            }
			
        }
    }*//*
    fclose(fTest);
	*/
	return 1;

};

int isLeaf::produceDictionary()
{
	
	char * filename = new char[100];		
	//to store the current input image
	Mat input;	

	//To store the keypoints that will be extracted by SIFT
	
	//To store the SIFT descriptor of current image
	
	//To store all the descriptors that are extracted from all the images.
	Mat featuresUnclustered;
	//The SIFT feature extractor and descriptor
		
	
	//I select 20 (1000/50) images from 1000 images to extract feature descriptors and build the vocabulary
	for(int f=1;f<291;f++){		
		//create the file name of an image
		sprintf(filename,"DictionaryCreation/1 (%i).jpg",f);
		//open the file
		input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale				
		//detect feature points
		detector.detect(input, keypoints);
		//compute the descriptors for each keypoint
		detector.compute(input, keypoints,descriptor);		
		//put the all feature descriptors in a single Mat object 
		featuresUnclustered.push_back(descriptor);		
		//print the percentage
		
	}	

	
	//Construct BOWKMeansTrainer
	//the number of bags
	int dictionarySize=200;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries=1;
	//necessary flags
	int flags=KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	Mat dictionary=bowTrainer.cluster(featuresUnclustered);	
	//store the vocabulary
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	
	system("PAUSE");
	return 0;

};


int isLeaf::produceTrainData()
{
	Mat dictionary;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"]>>dictionary;
	fs.release();
	Mat img;
	vector<KeyPoint> keypointT;
	Mat bowDescriptor;

	this->extractorT= Ptr<DescriptorExtractor>(new SiftDescriptorExtractor());
	this->matcherT= Ptr<DescriptorMatcher>(new FlannBasedMatcher());
	this->myBowT=Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractorT, matcherT));
	this->detectorT=new SiftFeatureDetector(500);

	myBowT->setVocabulary(dictionary);

	char filename[100];
	//To store the image tag name - only for save the descriptor in a file
	char * imageTag = new char[10];
	char * imageTag2 = new char[10];
	char * foldername= new char[100];
	

	DIR *dir;
	struct dirent *ent;
	char type;

	string str1="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Train/.";
	string str2="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Train/..";
	string str3="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Train/Thumbs.db";
	trainingData=Mat(0, 200, CV_32FC1);
	trainingDataLabels=Mat(0, 1, CV_32FC1);

	if ((dir = opendir ("C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Train/"))) 
	{
			while ((ent = readdir (dir)) != NULL) 
			{
				//FileStorage fs1("descriptor.yml", FileStorage::WRITE);	
				sprintf (filename, "C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Train/%s", ent->d_name);
				//the image file with the location. change it according to your image file location
				
				if(str1.compare(filename)==0 || str2.compare(filename)==0 || str3.compare(filename)==0)
					continue;
				
				type=ent->d_name[0];
				
				img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
				
				detectorT->detect(img, keypointT);

				myBowT->compute(img, keypointT, bowDescriptor);
				
				if(!bowDescriptor.empty())
				{
					trainingData.push_back(bowDescriptor);
					trainingDataLabels.push_back((float)type);
				}

				//prepare the yml (some what similar to xml) file
				//sprintf(imageTag,"img1");			
				//write the new BoF descriptor to the file
				//fs1 << imageTag << bowDescriptor;		
				//fs1.release();
				
				cout<<"Tip:"<<type<<endl;
				img.release();
			}
			closedir (dir);
	} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
			return -1;
		}

	FileStorage fs2("trainSet.yml", FileStorage::WRITE);	
	sprintf(imageTag,"trainSet");
	fs2 << imageTag << trainingData;		
	fs2.release();

	FileStorage fs3("trainSetLabels.yml", FileStorage::WRITE);
	sprintf(imageTag2, "trainLbl");
	fs3 << imageTag2 << trainingDataLabels;
	fs3.release();

	//Train

	CvSVM SVM;
	CvSVMParams params; 
    params.kernel_type=CvSVM::RBF; 
    params.svm_type=CvSVM::C_SVC; 
    params.gamma=0.50625000000000009; 
    params.C=312.50000000000000; 
    params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001); 

	cout<<"Train Basladi"<<endl;
	SVM.train_auto(trainingData,trainingDataLabels,Mat(), Mat(), params, 10); 
	cout<<"Train Bitti"<<endl;

	//End of train


	//Testing
	Ptr<DescriptorExtractor> extractorTest=Ptr<DescriptorExtractor>(new SiftDescriptorExtractor());
	Ptr<DescriptorMatcher> matcherTest=Ptr<DescriptorMatcher>(new FlannBasedMatcher());
	Ptr<BOWImgDescriptorExtractor> myBowTest=Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractorTest, matcherTest));
	Ptr<FeatureDetector> detectorTest=new SiftFeatureDetector(500);

	float response;
	int totalTestCounter=0;
	
	myBowTest->setVocabulary(dictionary);
	char filenameTest[100];

	DIR *dirTest;
	struct dirent *entTest;
	char typeTest;

	string str1Test="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/.";
	string str2Test="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/..";
	string str3Test="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/Thumbs.db";

	testingData=Mat(0, 200, CV_32FC1);
	testingDataLabels=Mat(0, 1, CV_32FC1);
	
	Mat imgTest;
	vector<KeyPoint> keypointTest;
	Mat bowDescriptorTest;

	if ((dirTest = opendir ("C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/"))) 
	{
			while ((entTest = readdir (dirTest)) != NULL) 
			{
				//FileStorage fs1("descriptor.yml", FileStorage::WRITE);	
				sprintf (filenameTest, "C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/%s", entTest->d_name);
				//the image file with the location. change it according to your image file location
				
				if(str1Test.compare(filenameTest)==0 || str2Test.compare(filenameTest)==0 || str3Test.compare(filenameTest)==0)
					continue;
				
				typeTest=entTest->d_name[0];
				
				imgTest=imread(filenameTest,CV_LOAD_IMAGE_GRAYSCALE);
				
				detectorTest->detect(imgTest, keypointTest);

				myBowTest->compute(imgTest, keypointTest, bowDescriptorTest);
				
				if(!bowDescriptorTest.empty())
				{
					testingData.push_back(bowDescriptorTest);
					testingDataLabels.push_back((float)typeTest);
				}

				//prepare the yml (some what similar to xml) file
				//sprintf(imageTag,"img1");			
				//write the new BoF descriptor to the file
				//fs1 << imageTag << bowDescriptor;		
				//fs1.release();
				
				cout<<"Tip:"<<typeTest<<endl;
				response=SVM.predict(bowDescriptorTest);
				cout<<"Response: "<<response<<endl;
				if(response==49)
				{
					totalTestCounter++;
				}
				imgTest.release();
			}
			closedir (dirTest);
	} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
			return -1;
		}


	
	cout<<"Bitti"<<endl;	
	cout<<"Yaprak sayisi: "<<totalTestCounter<<endl;
	system("PAUSE");
	return 0;

};

int isLeaf::produceTestData()
{
	
	Mat dictionary;
	FileStorage fs("dictionary.yml", FileStorage::READ);
	fs["vocabulary"]>>dictionary;
	fs.release();
	Mat img;
	vector<KeyPoint> keypointT;
	Mat bowDescriptor;

	this->extractorT= Ptr<DescriptorExtractor>(new SiftDescriptorExtractor());
	this->matcherT= Ptr<DescriptorMatcher>(new FlannBasedMatcher());
	this->myBowT=Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractorT, matcherT));
	this->detectorT=new SiftFeatureDetector(500);

	myBowT->setVocabulary(dictionary);

	char filename[100];
	//To store the image tag name - only for save the descriptor in a file
	char * imageTag = new char[10];
	char * imageTag2 = new char[10];
	char * foldername= new char[100];
	

	DIR *dir;
	struct dirent *ent;
	char type;

	string str1="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/.";
	string str2="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/..";
	string str3="C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/Thumbs.db";
	testingData=Mat(0, 200, CV_32FC1);
	testingDataLabels=Mat(0, 1, CV_32FC1);

	if ((dir = opendir ("C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/"))) 
	{
			while ((ent = readdir (dir)) != NULL) 
			{
				//FileStorage fs1("descriptor.yml", FileStorage::WRITE);	
				sprintf (filename, "C:/Users/EfeB/Documents/Visual Studio 2012/Projects/PlantRecognitionProject/x64/Release/Test/%s", ent->d_name);
				//the image file with the location. change it according to your image file location
				
				if(str1.compare(filename)==0 || str2.compare(filename)==0 || str3.compare(filename)==0)
					continue;
				
				type=ent->d_name[0];
				
				img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
				
				detectorT->detect(img, keypointT);
				myBowT->compute(img, keypointT, bowDescriptor);
				
				if(!bowDescriptor.empty())
				{
					testingData.push_back(bowDescriptor);
					testingDataLabels.push_back((float)type);
				}

				//prepare the yml (some what similar to xml) file
				//sprintf(imageTag,"img1");			
				//write the new BoF descriptor to the file
				//fs1 << imageTag << bowDescriptor;		
				//fs1.release();
				
				cout<<"Tip:"<<type<<endl;
				img.release();
			}
			closedir (dir);
	} 
		
	else {
			cout<<"Error exists"<<endl;
			perror ("");
			return -1;
		}

	FileStorage fs2("testSet.yml", FileStorage::WRITE);	
	sprintf(imageTag,"testSet");
	fs2 << imageTag << testingData;		
	fs2.release();

	FileStorage fs3("testSetLabels.yml", FileStorage::WRITE);
	sprintf(imageTag2, "testLbl");
	fs3 << imageTag2 << testingDataLabels;
	fs3.release();

	system("PAUSE");
	return 0;

};




void isLeaf::isLeafOrNot()
{
	
	Mat svmTrainData(377, 200, CV_32FC1);
	Mat svmTestData(277, 200, CV_32FC1);
	Mat svmTrainLabels(377, 1, CV_32FC1);
	Mat svmTestLabels(277, 1, CV_32FC1);
	
	CvMat cvTrainData;
	CvMat cvTestData;
	CvMat cvTrainLabels;
	CvMat cvTestLabels;

	CvSVM SVM;
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS,100,0.001);

	
	FileStorage fs("trainSet.yml", FileStorage::READ);
	fs["trainSet"]>>svmTrainData;
	fs.release();
	
	FileStorage fs1("trainSetLabels.yml", FileStorage::READ);
	fs1["trainLbl"]>>svmTrainLabels;
	fs1.release();

	FileStorage fs2("testSet.yml", FileStorage::READ);
	fs2["testSet"]>>svmTestData;
	fs2.release();

	FileStorage fs3("testSetLabels.yml", FileStorage::READ);
	fs3["testLbl"]>>svmTestLabels;
	fs3.release();
	

	float arrayTrainData[377][200];
	float arrayTestData[277][200];
	float arrayTrainLabels[377][1];
	float arrayTestLabels[277][1];
	
	
	for(int x=0; x<377; x++)
	{
		for(int y=0; y<200; y++)
		{
			arrayTrainData[x][y]=svmTrainData.at<float>(x,y);
		}
	}
	for(int x=0; x<377; x++)
	{
		for(int y=0; y<1; y++)
		{
			arrayTrainLabels[x][y]=svmTrainLabels.at<float>(x,y);
		}
	}
	for(int x=0; x<277; x++)
	{
		for(int y=0; y<200; y++)
		{
			arrayTestData[x][y]=svmTestData.at<float>(x,y);
		}
	}
	for(int x=0; x<277; x++)
	{
		for(int y=0; y<1; y++)
		{
			arrayTestLabels[x][y]=svmTestLabels.at<float>(x,y);
		}
	}
	cvInitMatHeader(&cvTrainData,377,200,CV_32FC1,arrayTrainData); //eðitim verimizi tutacak matris.
	cvInitMatHeader(&cvTrainLabels,377,1,CV_32SC1,arrayTrainLabels);
	cout<<"Train started:"<<endl;
	//SVM.train(&cvTrainData,&cvTrainLabels,0,0,CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,3,0.025,0,1,0.5,0.1,NULL,criteria));
	cout<<"Train ended"<<endl;
	
	
	/*
	CvSVM SVM;
	CvMat trainData;
	CvMat testData;
	CvMat labels;
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS,100,0.001);
	float* arrayTrainData=(float*)myTrainData.data;
	float* arrayTrainDataLabel=(float*)myTrainDataLabel.data;
	float* arrayTestData=(float*)myTestData.data;

	for(int a=0; a<myTrainDataLabel.rows; a++)
		cout<<arrayTrainDataLabel[a];

	#pragma endregion data
	cvInitMatHeader(&trainData,Y_NUMBER_OF_TRAINING_SAMPLES,NUMBER_OF_ATTRIBUTES,CV_32FC1,arrayTrainData); //eðitim verimizi tutacak matris.
	cvInitMatHeader(&labels,1,Y_NUMBER_OF_TRAINING_SAMPLES,CV_32SC1,arrayTrainDataLabel); //etiketlerimizi tutacak matris.
            //classification olduðu için CV_32SC1 kullandýk.
	cvInitMatHeader(&testData,Y_NUMBER_OF_TEST_SAMPLES,NUMBER_OF_ATTRIBUTES,CV_32FC1,arrayTestData);
	
	SVM.train(&trainData,&labels,0,0,CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,3,0.025,0,1,0.5,0.1,NULL,criteria));  
	  //train fonksiyonumuz (libsvm ile kontrol etmek için libsvm ile
      //gelen svm-train.exe nin default parametreleri kullanýldý.)
      //Parametrelerin detaylý açýklamasý için opencv docs.
	CvMat results;
	//SVM.predict(arrayTestData, &results);
	float sonuc=0;
	int sayici=0;
	for(int i = 0; i < Y_NUMBER_OF_TEST_SAMPLES; i++ )
	{
		double r;
		CvMat sample;
		cvGetRow(&testData, &sample, i );
		r = SVM.predict( &sample);  //predict edip sonuçlarý kontrol edelim.
		if(r==0)
			sayici++;

	}
	cout<<"Sifir sayisi: "<<sayici<<endl;	
	*/

};
