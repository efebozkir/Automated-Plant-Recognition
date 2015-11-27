/*
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

#include "ClassifyIndividual.h"
#include "ExtractDescriptorHelper.h"

using namespace std;
using namespace cv;


void ClassifyIndividual::sortDataVect()
{
	DiffvName temp;
    bool finished = false;
    while (!finished)    {
       finished = true;
       for (int i = 0; i < myVect.size()-1; i++) {
		   if (myVect[i].difference > myVect[i+1].difference) {
             temp = myVect[i];
             myVect[i] = myVect[i+1];
             myVect[i+1] = temp;
             finished=false;
          }
        }
     }

};

void ClassifyIndividual::printVector()
{
	for(int a=0; a<10; a++)
	{
		
		cout<<a<<". name: "<<myVect.at(a).name<<endl;
		cout<<a<<". difference"<<myVect.at(a).difference<<endl;
	}
}

int ClassifyIndividual::makeClassification()
{
	DIR *dir;
	struct dirent *ent;
	char folder[100];
	string topTenNames[10];
	double topTenDifference[10];

	for(int m=0; m<9; m++)
		topTenDifference[m]=1000;

	string str1="C:/Features/.";
	string str2="C:/Features/..";
	
	DiffvName dataStruct;


	IplImage* src;
	src=cvLoadImage("C:/TestSet/Abutilon_theophrasti-4.png");
	ExtractDescriptorHelper* tmpDescriptorFinder=new ExtractDescriptorHelper();
	tmpDescriptorFinder->ExtractDescriptors(src);
					
	tmpDescriptorFinder->sortVector();
	tmpDescriptorFinder->createFeatureVector();

	for(int p=0; p<59; p++)
		compareTestSetArray[p]=tmpDescriptorFinder->getMyFeatureVector().at(p);


	
	if ((dir = opendir ("C:/Features/"))) 
		{
			
			while ((ent = readdir (dir)) != NULL) 
			{
					dataStruct.difference=0;
					dataStruct.name="x";

					sprintf (folder, "C:/Features/%s", ent->d_name);
					//cout<<"File name:"<<folder<<endl;
					
					if(str1.compare(folder)==0 || str2.compare(folder)==0)
						continue;

					ifstream myfile;
					
					myfile.open(folder);
						for(int i=0; i<59; i++)
						{
							myfile>>compareTrainSetArray[i];
							//cout << compareTrainSetArray[i] << endl;
							
						}

				

					double tmpSum=0;
					for(int x=0; x<59;x++)
					{
						tmpSum=tmpSum+pow((compareTrainSetArray[x]-compareTestSetArray[x]),2);
					}
					tmpSum=sqrt(tmpSum);

					//cout<<"Sayi:"<<tmpSum<<endl;
					
					dataStruct.difference=tmpSum;
					dataStruct.name=ent->d_name;
					
					myVect.push_back(dataStruct);
					
								
					
					
			}
		closedir (dir);
		return 0;
		} 
		
		else {
			
			cout<<"Error exists"<<endl;
			perror ("");
			return EXIT_FAILURE;
			}


		


};
*/



	/*
					ofstream myfile;
					myfile.open (writeFile);
					for(int k=0; k<59; k++)
						myfile << featureArray[k]<<"\n";
					myfile.close();*/