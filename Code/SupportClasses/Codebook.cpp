#include "Codebook.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <highgui.hpp>
#include <imgcodecs.hpp>

Codebook::Codebook(int codebookSize, int descriptorSize, std::vector<cv::Mat>& positiveExamplesDescriptors) {
	mCodebookSize = codebookSize;
	mDescriptorSize = descriptorSize;

	// Loading the codebook images
	std::vector<cv::Mat> codebookImages;
	loadImages(codebookImages, "codebook images");

	// Extracting for each image the boats highlighted in the ground truth coordinates
	std::vector<cv::Mat> boatExamples;
	std::vector<std::vector<int>> gtCoordinates(codebookImages.size());
	extractBoatExamples(codebookImages, boatExamples, gtCoordinates);

	// Computing for each boat its SIFT descriptors and returning all the descriptors in a single Nx128 matrix
	std::vector<cv::Mat> boatDescriptors(boatExamples.size());
	cv::Mat allBoatsDescriptors = computeFeatureDescriptors(boatExamples, boatDescriptors, gtCoordinates);
	positiveExamplesDescriptors = boatDescriptors;

	// Clustering all the descriptors
	performKMeans(allBoatsDescriptors, mCodebook);
}

Codebook::Codebook(int codebookSize, int descriptorSize, std::string existingCodebook) {
	mCodebookSize = codebookSize;
	mDescriptorSize = descriptorSize;
	mCodebook = readCodebookFromTxt(existingCodebook);
}

cv::String Codebook::selectCodebook(){
	bool stop = true;
	cv::String selection;
	while(stop){
		std::cout << "Do you want to generate a new codebook from scratch [N] or use the predefined one [P]?"<<std::endl;
		std::cin >> selection;
		if (selection.compare("N") != 0 && selection.compare("P")!=0) {
			std::cout << "Error: type N for a new codebook or P for the predefined one." <<std::endl;
			std::cin.clear();
			std::cin.ignore(10000, '\n');
		}else{
			stop = false;
		}
	}
	return selection;
}

void Codebook::loadImages(std::vector<cv::Mat>& images, std::string s){
	std::cout<<"Please insert the directory containing the "<<s<<": "<<std::endl;
	cv::String directory;
	std::cin >> directory;
	if (checkPath(directory) == 0) {
		std::cout << "Error: folder \"" << directory << "\" not present." << std::endl;
		exit(1);
	}
	std::cout << "Loading all the "<<s<<"..." << std::endl;
	std::vector<cv::String> pattern = {"*.png", "*.jpg", "*.jpeg", "*.bmp"};
	std::vector<cv::String> imagesNames;
	try {
		int i = 0;
		while (i<pattern.size()) {
			cv::utils::fs::glob(directory, pattern[i], imagesNames);
			i++;
		}
	} catch (const cv::Exception& e) {
		std::cerr << "Attention! Exception occurred in loading the images names: \n" << e.msg << std::endl;
		exit(1);
	}
	cv::Mat currImage;
	for(int i=0; i<imagesNames.size(); i++) {
		currImage = cv::imread(imagesNames[i]);
		images.push_back(currImage);
	}
	if (images.size()==0){
		std::cout<<"Error: the folder specified doesn't contains images. Please provide a folder containing some images."<<std::endl;
		exit(1);
	}
	std::cout << "Loading completed: loaded " <<images.size()<< " images."<< std::endl;
}

bool Codebook::checkPath(const std::string &s){
	struct stat buffer{};
	return (stat (s.c_str(), &buffer) == 0);
}

void Codebook::readGroundTruthCoordsFromTxT(std::vector<cv::Mat> codebookImages, std::vector<std::vector<int>>& gtCoordinates){
	bool stop;
	std::string txtFileName;
	std::ifstream txtFile;
	stop = false;
	while(!stop){
		std::cout<<"Please insert the path of the .txt file containing the ground truth coordinates of the boats:"<<std::endl;
		std::cin >> txtFileName;
		txtFile.open(txtFileName);
		if (!txtFile.is_open()){
			std::cout<<"Error: the file \""<<txtFileName<<"\" cannot be found"<<std::endl;
		} else
			stop = true;
	}
	std::cout<<"Extraction of the boats examples..."<<std::endl;
	char c;
	std::vector<int> nBoats(codebookImages.size());
	std::vector<std::vector<int>> xyCoords(codebookImages.size());
	int imgIdx = 0;
	while (1){
		//std::cout<<"----------    Image ["<<imgIdx<<"]    ----------"<<std::endl;
		int spaces = 0;
		int k = 0;
		std::string s  = "";
		c = txtFile.get();
		bool goOn = true;
		while (goOn){
			if (c!=' ')
				s = s+c;
			else{
				if (spaces == 0)
					s = "";
				if (spaces == 1){
					nBoats[imgIdx] = std::stoi(s);
					//std::cout << "nBoats["<<imgIdx<<"] = "<<nBoats[imgIdx]<<std::endl;
					s = "";
				}
				if (spaces > 1){
					xyCoords[imgIdx].push_back(std::stoi(s));
					//std::cout << "xyCoords[" << imgIdx << "]["<<k<<"] = " << std::stoi(s)<< std::endl;
					k++;
					s = "";
				}
				spaces = spaces + 1;
			}
			if (c=='\n'){
				xyCoords[imgIdx].push_back(std::stoi(s));
				//std::cout << "xyCoords[" << imgIdx << "]["<<k<<"] = " << std::stoi(s)<< std::endl;
				k++;
				s = "";
				goOn = false;
			}
			c = txtFile.get();
			//std::cout << c;
		}
		std::vector<int> currWindowCorners(xyCoords[imgIdx].size());
		currWindowCorners = xyCoords[imgIdx];
		for (int i = 0; i < 2*2*nBoats[imgIdx]-1; i=i+4) {
			//std::cout<<"currWindowsCorners ["<<i<<"] = "<<currWindowCorners[i]<<std::endl;
			//std::cout<<"currWindowsCorners ["<<i+1<<"] = "<<currWindowCorners[i+1]<<std::endl;
			int x = currWindowCorners[i];
			int y = currWindowCorners[i+1];
			int w = currWindowCorners[i+2];
			int h = currWindowCorners[i+3];
			gtCoordinates[imgIdx].push_back(x);
			gtCoordinates[imgIdx].push_back(y);
			gtCoordinates[imgIdx].push_back(w);
			gtCoordinates[imgIdx].push_back(h);
			//std::cout<<"TopLeftCorner["<<i/4<<"] = ["<<x<<", "<<y<<"]"<<std::endl;
			//std::cout<<"[width, height]["<<i/4<<"] = ["<<w<<", "<<h<<"]"<<std::endl;
		}
		imgIdx = imgIdx + 1;
		//std::cout<<"txtFile.eof() = "<<txtFile.eof()<<std::endl;
		if (txtFile.eof()){
			//std::cout<<"EOF"<<std::endl;
			txtFile.close();
			break;
		}
	}
}

void Codebook::extractBoatExamples(std::vector<cv::Mat> codebookImages, std::vector<cv::Mat>& boatExamples, std::vector<std::vector<int>>& gtCoordinates){
	readGroundTruthCoordsFromTxT(codebookImages, gtCoordinates);
	for (int i = 0; i < gtCoordinates.size(); i++) {
		for (int j = 0; j < gtCoordinates[i].size(); j=j+4) {
			//std::cout<<"Image "<<i<<" window "<<j/4<<std::endl;
			int xTopLeftCorner = gtCoordinates[i][j];
			int yTolLeftCorner = gtCoordinates[i][j+1];
			int width = gtCoordinates[i][j+2];
			int height = gtCoordinates[i][j+3];
			cv::Rect roi(xTopLeftCorner,yTolLeftCorner, width, height);
			cv::Mat currBoatExample(codebookImages[i](roi).rows, codebookImages[i](roi).cols, CV_16F);
			codebookImages[i](roi).copyTo(currBoatExample);
//			cv::namedWindow("a",cv::WINDOW_NORMAL);
//			cv::imshow("a",currBoatExample);
//			cv::waitKey(0);
			boatExamples.push_back(currBoatExample);
		}
	}
	std::cout<<"Extraction complete: extracted " <<boatExamples.size()<<" examples (sub-images)."<<std::endl;
}

cv::Mat Codebook::computeFeatureDescriptors(std::vector<cv::Mat>& examples, std::vector<cv::Mat>& descriptors, std::vector<std::vector<int>>& gtCoordinates){
	std::cout<<"Computation of the features descriptors..."<<std::endl;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0,3,0.04,10,1.6);
	std::vector<std::vector<cv::KeyPoint>> keypoints(examples.size());
	std::cout<<"Example ";
	std::vector<cv::Mat> examplesWithFeaturesForDebug = examples;
	for (int i = 0; i < descriptors.size(); i++) {
		std::string s = std::to_string(i+1) + "/" + std::to_string(descriptors.size());
		std::cout<<s;
		std::cout << std::string(s.length(),'\b');
		detector->detectAndCompute(examples[i],cv::Mat(), keypoints[i], descriptors[i], false);
		//std::cout<<"keypoints["<<i<<"].size() = "<<keypoints[i].size()<<std::endl;
		//std::cout<<"descriptors["<<i<<"].size() = ["<<descriptors[i].rows<<", "<<descriptors[i].cols<<"]"<<std::endl;
		cv::drawKeypoints(examples[i],keypoints[i],examplesWithFeaturesForDebug[i],cv::Scalar(0,255,0));
	}
	std::cout<<"\n";
	cv::Mat allDescriptors;
	cv::vconcat(descriptors, allDescriptors);
	std::cout<<"Computation completed: allDescriptors.size() = ["<<allDescriptors.rows<<", "<<allDescriptors.cols<<"]"<<std::endl;

	// Just for debugging
	printPositiveExamplesForDebug(examplesWithFeaturesForDebug, gtCoordinates);
	return allDescriptors;
}

void Codebook::performKMeans(cv::Mat allDescriptors, cv::Mat& centroids){
	cv::Mat allDescriptors32F;
	cv::Mat labels;
	allDescriptors.convertTo(allDescriptors32F,CV_32F);
	std::cout<<"Performing the k-means clustering [using the all the "<<allDescriptors.rows<<" descriptors]: please be patient it could take a few minutes..."<<std::endl;
	//cv::Rect roi(0,0, mDescriptorSize,allDescriptors.rows);
	time_t tStart, tEnd;
	tStart = time(0);
	cv::kmeans(allDescriptors32F, mCodebookSize, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 10, 3.0), 1, cv::KMEANS_PP_CENTERS, centroids);
	tEnd = time(0);
	std::cout<<"Clustering completed in "<< static_cast<int>(difftime(tEnd,tStart))/60 <<"m "<< int(difftime(tEnd,tStart)) % 60 <<"s"<<std::endl;
	writeMatToTxt(centroids, "centroids.txt");
	std::cout<<"Codebook generated: check out the file \"centroids.txt\"."<<std::endl;
}

void Codebook::writeMatToTxt(cv::Mat M, std::string str){
	std::ofstream outTxt(str);
	for (int i = 0; i < M.rows; i++) {
		for (int j = 0; j < M.cols; j++) {
			outTxt << M.at<float>(i,j);
			if (j<M.cols-1)
				outTxt<<" ";
		}
		outTxt<<"\n";
	}
	outTxt.close();
}

cv::Mat Codebook::readCodebookFromTxt(std::string str){
	std::ifstream txtFile;
	txtFile.open(str);
	if (!txtFile.is_open()){
		std::cout<<"Error: the precomputed coodbook \""<<str<<"\" cannot be found. "<<std::endl;
		exit(1);
	}
	std::cout<<"Loading the precomputed codebook..."<<std::endl;
	std::vector<std::vector<float>> v(mCodebookSize);
	int i = 0;
	std::string s;
	char c = txtFile.get();
	int k = 0; //For debugging
	while(1){
		bool goOn = true;
		while (goOn){
			if (c!=' ' && c!='\n')
				s = s+c;
			else {
				v[i].push_back(std::stof(s));
				//std::cout<<"v["<<i<<"]["<<k<<"] = "<<std::stof(s)<<std::endl;
				k++;
				s = "";
				if (c == '\n') {
					goOn = false;
				}
			}
			c = txtFile.get();
		}
		k=0;
		i++;
		if (txtFile.eof()){
			txtFile.close();
			break;
		}
	}
	std::cout<<"Codebook loaded."<<std::endl;
	return convert2cvMat(v);
}

cv::Mat Codebook::convert2cvMat(std::vector<std::vector<float>> v){
	//std::cout<<"codebook converted"<<std::endl;
	cv::Mat M(mCodebookSize, mDescriptorSize, CV_32F);
	for (int i = 0; i < mCodebookSize; i++) {
		for (int j = 0; j < mDescriptorSize; j++) {
			M.at<float>(i,j) = v[i][j];
		}
	}
	//std::cout<<"codebook converted"<<std::endl;
	return M;
}

cv::Mat Codebook::returnCodebook(){
	return mCodebook;
}

int Codebook::returnCodebookSize(){
	return mCodebookSize;
}

int Codebook::returnDescriptorSize(){
	return mDescriptorSize;
}

void Codebook::printPositiveExamplesForDebug(std::vector<cv::Mat> inputImages, std::vector<std::vector<int>> gtCoordinates){
	for (int i = 0; i < inputImages.size(); i++) {
//		for (int j = 0; j < gtCoordinates[i].size(); j=j+4) {
//			cv::Point2f topLeftCorner(gtCoordinates[i][j], gtCoordinates[i][j+1]);
//			cv::Point2f bottomRightCorner(gtCoordinates[i][j]+gtCoordinates[i][j+2], gtCoordinates[i][j+1]+gtCoordinates[i][j+3]);
//			cv::rectangle(inputImages[i], topLeftCorner, bottomRightCorner,cv::Scalar(0,255,0), 2);
//		}
		cv::imwrite("checkPositiveExamples\\img"+std::to_string(i)+".jpg", inputImages[i]);
	}
}

