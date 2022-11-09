#include <cmath>
#include <iostream>
#include <highgui.hpp>
#include <imgcodecs.hpp>
#include "sys/stat.h"
#include "SupportClasses/Codebook.h"
#include "SupportClasses/Graph.h"
#include <imgproc.hpp>
#include <ml.hpp>
#include <fstream>
#include <ostream>
#include <iomanip>
#include <list>
#include <iterator>
#include "imgproc.hpp"

// FUNCTIONS DECLARATIONS
std::string selector(int caseParam);
void loadImages(std::vector<cv::Mat>& images, std::string s);
cv::Mat computeFeatureDescriptors(std::vector<cv::Mat>& examples, std::vector<cv::Mat>& descriptors);

void loadInputImage(cv::Mat& inputImage);
bool checkPath(const std::string &s);
void readGroundTruthCoordsFromTxT(std::vector<cv::Mat> codebookImages, std::vector<std::vector<int>>& gtCoordinates);
void extractBoatExamples(std::vector<cv::Mat> codebookImages, std::vector<cv::Mat>& boatExamples, std::vector<std::vector<int>>& gtCoordinates);

int getNearestDivisibleInt(int stepSize, int length);
void computeFeatureBlocks(cv::Mat inputImage, int windowSide, int stepSize, std::vector<std::vector<cv::KeyPoint>>& allKeypoints, std::vector<cv::Mat>& allDescriptors);
void initializeWindowsBlocks(std::list<int>& windowBlockIndeces, int windowSide, int stepSize, int n, int i, int j);
void updateCurrWindowsBlocks(std::list<int>& windowBlockIndeces, int shift, int windowSide, int stepSize, int n, int i, int j);
cv::Mat returnFeaturesOfCurrWindow(std::list<int> windowBlockIndeces,  std::vector<std::vector<cv::KeyPoint>> allKeypoints, std::vector<cv::Mat> allDescriptors,
								   std::vector<std::vector<cv::KeyPoint>>& currKeypoints, std::vector<cv::Mat>& currDescriptors, int descriptorSize, int r, int c);
void fromVecOfVec2Vec(std::vector<std::vector<cv::KeyPoint>> vecOfVec, std::vector<cv::KeyPoint>& vec);
void personalVConcat(std::vector<cv::Mat> matrices, cv::Mat& globMatrix);
void takeInputsWindowSideAndStepSize(int& windowSide, int& stepSize);
void drawRectangles(bool drawAll, int l, int i, int j, cv::Mat& inputImageCopyForDrawing, int stepSize, int windowSide, cv::Scalar color);
cv::Mat returnBoWFromDescriptors(cv::Mat descriptors, Codebook c, int windowNumber, double ratioThreshold, double distanceThreshold);
cv::Mat returnBoWsOfTheExamples( std::vector<cv::Mat> descriptors, Codebook c, std::string s);
double returnFractionOfOverlappedArea(int i1, int j1, int i2, int j2, int l);
std::vector<std::vector<int>> returnIntermediateBoxes(std::vector<cv::Point2f> currImgTopLeftCorners, int l, double minOverlap);
void refineBox(int i, int j, int w, int h, double& xMin, double& yMin, double& wNew, double& hNew, int n, int stepSize, std::vector<std::vector<cv::KeyPoint>> allKeypoints);
void resizeBox(double& xMin, double& yMin, double& w, double& h, int resizeFactor);
double returnAbsoluteOverlappedArea(double xMin1, double yMin1, double w1, double h1, double xMin2, double yMin2, double w2, double h2);
std::vector<std::vector<double>> returnFinalBoxes(std::vector<std::vector<double>> boxes, double minOverlap);
void computeBoWsOfBlocks(Codebook c, double ratioThreshold, double distThreshold, std::vector<cv::Mat> _allDescriptors, std::vector<cv::Mat>& BoWsOfAllBlocks);
cv::Mat returnBoWOFCurrentWindow(std::list<int> windowBlockIndeces, std::vector<cv::Mat> BoWsOfAllBlocks);
void pruneBoxes(std::vector<std::vector<double>>& boxes, double overlappingRatio);
void saveWindow(int key, int& wait, int i, int j, int windowSide, int stepSize, cv::Mat inputImage);
std::vector<std::vector<double>> extractNotMergedBoxes(std::vector<std::vector<double>> boxes, int level);
std::vector<std::vector<double>> extractFinalSameLevelNotMergedBoxes(std::vector<std::vector<double>> finalBoxesOfAllLayers, int level);
void readInputImageGTCoords(std::vector<std::vector<double>>& GTCoords, double wRatio, double hRatio);
bool liesInsideABox(std::vector<double> point, std::vector<std::vector<double>> boxes);
std::vector<std::vector<std::vector<double>>> returnUnionPoligon(std::vector<std::vector<double>> finalBoxes, std::vector<std::vector<double>> GTCoords);
void sortNodes(std::vector<std::vector<std::vector<double>>>& poligonsNodes);
std::vector<std::vector<double>> returnIntersectionPoints(std::vector<std::vector<double>> overlappingBoxes);
void removeDuplicates(std::vector<std::vector<double>>& points);
std::vector<double> getMinDistPoint(std::vector<double> currPoint, std::vector<std::vector<double>> candidates);
int returnIndex(std::vector<double> point, std::vector<std::vector<double>> pointsVec);
int getTopLeftNodeIdx(std::vector<std::vector<double>> pointsVec);
void removeDuplicates(std::vector<std::vector<std::vector<double>>>& allPoints);
void moveAhead(std::vector<double>& currNode, int i, std::vector<int>& visited, std::vector<std::vector<std::vector<double>>>& tmpNodes,
			   int& currIdx, std::vector<std::vector<std::vector<double>>> poligonsNodes, std::vector<std::vector<double>> candidates, int& lastMovement,
			   int currMovement, bool& addedANode);
double returnArea(std::vector<std::vector<std::vector<double>>> polygon);
std::vector<std::vector<std::vector<double>>> returnIntersectionPoligon(std::vector<std::vector<double>> finalBoxes, std::vector<std::vector<double>> GTCoords);

// Debug section
void printImageFeatureBlocks(std::vector<std::vector<cv::KeyPoint>> allKeypoints, std::vector<cv::Mat> allDescriptors, int n, int stepSize);
void printListElements(std::list<int> list);
void printWindowFeatureBlocks(std::vector<std::vector<cv::KeyPoint>> currKeypoints, std::vector<cv::Mat> currDescriptors, int i, int j, int n, int stepSize);
void printDescriptors(std::string s, cv::Mat descriptors);
void showBoW(cv::Mat inputBoW, int BoWnum);

// START MAIN
int main(int argc , char** argv) {
	std::cout<<"Do you want to try the offline phase (codebook generation and/or SVM training) [0] or the online phase (boat detection) [1]? ";
	std::string phase = selector(0);
	int codebookSize = 12500;
	int descriptorSize = 128;
	if (phase.compare("0") == 0) {
		std::cout << "Selected offline phase." << std::endl;
		std::cout << "Do you want to generate a new codebook?[y/n] ";
		std::string option1 = selector(1);
		if (option1.compare("y") == 0) {
			// Starting the generation of a new codebook
			std::cout << "=======================================        GENERATION OF A NEW CODEBOOK        =======================================" << std::endl;
			std::vector<cv::Mat> positiveExampleDescriptors;
			Codebook c(codebookSize, descriptorSize, positiveExampleDescriptors);
		}
		std::cout << "Do you want to train the SVM?[y/n]: ";
		std::string option2 = selector(1);
		if (option2.compare("y") == 0) {
			// Starting the training of a SVM
			std::cout <<"\n=======================================          TRAINING OF THE SVM              =======================================" << std::endl;
			// Loading a precomputed codebook
			Codebook c(codebookSize, descriptorSize, "codebookCentroids.txt");

			// Loading the positive examples
			std::vector<cv::Mat> multiplePositiveExamples;
			loadImages(multiplePositiveExamples, "positive examples");

			// Extracting for each image the boats highlighted in the ground truth coordinates
			std::vector<cv::Mat> positiveExamples;
			std::vector<std::vector<int>> gtCoordinates(multiplePositiveExamples.size());
			extractBoatExamples(multiplePositiveExamples, positiveExamples, gtCoordinates);

			// Computing the SIFT descriptor of each positive example
			std::vector<cv::Mat> positiveExamplesDescriptors(positiveExamples.size());
			computeFeatureDescriptors(positiveExamples, positiveExamplesDescriptors);
			std::cout<<"positiveExamplesDescriptors.size() = "<<positiveExamplesDescriptors.size()<<std::endl;

			// Computing the BoW of each positive example wrt to the previous codebook
			cv::Mat positiveBoWs = returnBoWsOfTheExamples(positiveExamplesDescriptors, c, "positive examples");
			std::cout<<"positiveBoWs.rows = "<<positiveBoWs.rows<<std::endl;

			// Loading the negative examples
			std::vector<cv::Mat> negativeExamples;
			loadImages(negativeExamples, "negative examples");

			// Computing the SIFT descriptor of each negative examples
			std::vector<cv::Mat> negativeExamplesDescriptors(negativeExamples.size());
			computeFeatureDescriptors(negativeExamples, negativeExamplesDescriptors);

			// Computing the BoW of each negative example wrt to the previous codebook
			cv::Mat negativeBoWs = returnBoWsOfTheExamples(negativeExamplesDescriptors, c, "negative examples");
			std::cout<<"negativeBoWs.rows = "<<negativeBoWs.rows<<std::endl;

			// Launching the training of the SVM
			cv::Mat trainingData;
			cv::vconcat(positiveBoWs, negativeBoWs, trainingData);
			std::cout << "trainingData.size() = [" << trainingData.rows << ", " << trainingData.cols << "]" << std::endl;
			cv::Mat positiveLabels(positiveBoWs.rows, 1, CV_32SC1, cv::Scalar::all(1));
			cv::Mat negativeLabels(negativeBoWs.rows, 1, CV_32SC1, cv::Scalar::all(0));
			cv::Mat labels;
			cv::vconcat(positiveLabels, negativeLabels, labels);
			std::cout << "labels.size() = [" << labels.rows << ", " << labels.cols << "]" << std::endl;

			std::cout << "Launch training of the SVM..." << std::endl;
			cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

			svm->setType(cv::ml::SVM::C_SVC);
			svm->setKernel(cv::ml::SVM::RBF);
			svm->setNu(0.5);
			svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 500, 1e-6));
			time_t tStart, tEnd;
			tStart = time(0);
			svm->trainAuto(cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, labels));
			tEnd = time(0);
			std::cout<<"Training completed in "<< static_cast<int>(difftime(tEnd,tStart))/60 <<"m "<< int(difftime(tEnd,tStart)) % 60 <<"s"<<std::endl;

			svm->save("pretrainedSvm.xml");
		}
	} else {
		std::cout << "Selected online phase.\n" << std::endl;

		// Loading a precompbauted codebook
		Codebook c(codebookSize, descriptorSize, "codebookCentroids.txt");

		// Loading a pretrained SVM
		std::cout << "\nLoading a pretrained SVM..." << std::endl;
		cv::Ptr<cv::ml::SVM> svm = cv::Algorithm::load<cv::ml::SVM>("pretrainedSvm.xml");
		std::cout << "SVM loaded.\n" << std::endl;

		// Fixing window side, step size and new image side
		int windowSide = 100;
		int stepSize = 10;


		// Loading the input image
		cv::Mat inputImage;
		loadInputImage(inputImage);

		int N;
		int pyramidLevels;
		if (inputImage.cols > 600 && inputImage.rows >600){
			N = 800;
			pyramidLevels = 3;
		} else if (inputImage.cols > 300 && inputImage.rows >300) {
			N = 400;
			pyramidLevels = 2;
		} else {
			N = 200;
			pyramidLevels = 1;
		}
		std::cout<<"Pyramid levels = "<<pyramidLevels<<std::endl;
		int M = N;

		// Saving aspects ratio of the original image for converting correctly the GT coordinates
		double wRatio = (static_cast<double>(N))/(static_cast<double>(inputImage.cols));
		double hRatio = (static_cast<double>(N))/(static_cast<double>(inputImage.rows));

		// Reading GT coordinates
		std::vector<std::vector<double>> GTCoords;
		readInputImageGTCoords(GTCoords, wRatio, hRatio);

		// Performing the preliminary resizing
		cv::resize(inputImage, inputImage, cv::Size(N, M), cv::INTER_LINEAR);

		// Defining some copies of the input images for drawing
		cv::Mat inputImageAllScaleBoxes;
		cv::Mat inputImageTemporaryResult;
		cv::Mat inputImageFinalResult;
		inputImage.copyTo(inputImageAllScaleBoxes);
		inputImage.copyTo(inputImageTemporaryResult);
		inputImage.copyTo(inputImageFinalResult);

		std::cout<<"\nPerforming a preliminary bilinear smoothing..."<<std::endl;
		cv::Ptr<cv::SIFT> prelimDetector = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
		std::vector<cv::KeyPoint> prelimAllKeypoints;
		cv::Mat prelimAllDescriptors;
		prelimDetector->detectAndCompute(inputImage, cv::Mat(), prelimAllKeypoints, prelimAllDescriptors, false);
		int nKeypoints = prelimAllKeypoints.size();
		int diameter;
		int p1;
		int p2;
		if (nKeypoints<=7000) {
			diameter = nKeypoints / 43;
			p1 = nKeypoints / 43;
			p2 = nKeypoints / 43;
		}else if (nKeypoints>7000){
			diameter = nKeypoints / 32;
			p1 = nKeypoints / 32;
			p2 = nKeypoints / 32;
		}

		cv::Mat tmpImage;
		cv::bilateralFilter(inputImage, tmpImage, diameter, p1, p2);
		tmpImage.copyTo(inputImage);
		std::cout<<"Smoothing complete.\n"<<std::endl;

		std::vector<std::vector<cv::Point2f>> topLeftCorners(pyramidLevels);
		std::vector<std::vector<double>> finalBoxesNotMergedLayer0;
		std::vector<std::vector<double>> finalBoxesNotMergedLayer1;
		std::vector<std::vector<double>> finalBoxesNotMergedLayer2;
		std::vector<std::vector<double>> finalBoxesOfAllLayers;
		int stepSizeCopy = stepSize;
		for (int s = 0; s < pyramidLevels; s++) {
			std::cout<<"---------------   Starting the research in the level "<<s+1<<" of the image pyramid.   ---------------"<<std::endl;

			int n = N / stepSize;
			int m = M / stepSize;


			std::vector<std::vector<cv::KeyPoint>> _allKeypoints(n * m); //_allKeypoints plays the role of \mathbf{v} in the report
			std::vector<cv::Mat> _allDescriptors(n * m);

			computeFeatureBlocks(inputImage, windowSide, stepSize, _allKeypoints, _allDescriptors);

			double ratioThreshold = 0.95;
			double distanceThreshold = 1000;

			std::vector<cv::Mat> BoWsOfAllBlocks(_allKeypoints.size());
			computeBoWsOfBlocks(c, ratioThreshold, distanceThreshold, _allDescriptors, BoWsOfAllBlocks);

			int l = windowSide / stepSize;
			std::list<int> windowBlockIndeces;

			// Sliding window procedure
			cv::Mat tmpInputImage;
			inputImage.copyTo(tmpInputImage);

			cv::Ptr<cv::SIFT> _detector = cv::SIFT::create(0, 3, 0.04, 10, 1.6);

			// JUST FOR DEBUGGING
			std::vector<cv::KeyPoint> allKeypointsForDrawing;
			cv::Mat allDescriptorsForDrawing;
			_detector->detectAndCompute(tmpInputImage, cv::Mat(), allKeypointsForDrawing, allDescriptorsForDrawing, false);
			cv::drawKeypoints(inputImage, allKeypointsForDrawing, tmpInputImage, cv::Scalar(0, 128, 255));

			std::vector<std::vector<cv::KeyPoint>> _currKeypoints(l * l);
			std::vector<cv::Mat> _currDescriptors(l * l);

			std::vector<cv::KeyPoint> currKeypointsForDrawing;
			cv::Mat inputImageCopyForDrawing2;
			tmpInputImage.copyTo(inputImageCopyForDrawing2);

			int rightShifts = (N - windowSide) / stepSize + 1;
			int downShifts = (M - windowSide) / stepSize + 1;

			std::list<int> supportIndeces;
			cv::Mat currDescriptorsTogether;
			cv::Mat imageForDrawing;
			inputImageCopyForDrawing2.copyTo(imageForDrawing);

			std::cout<<"Performing the sliding window procedure..."<<std::endl;
			int wait = 1;
			for (int i = 0; i < downShifts; i++) {
				for (int j = 0; j < rightShifts; j++) {
					if (i == 0 && j == 0) { // initialization + definition of support indeces
						initializeWindowsBlocks(windowBlockIndeces, windowSide, stepSize, n, i, j);
						supportIndeces = windowBlockIndeces;
						currDescriptorsTogether = returnFeaturesOfCurrWindow(windowBlockIndeces, _allKeypoints, _allDescriptors, _currKeypoints, _currDescriptors, c.returnDescriptorSize(), i, j);
						//printWindowFeatureBlocks(_currKeypoints, _currDescriptors, i, j, l, stepSize);
					} else if (i == 0 && j > 0) { // right shift [0]
						updateCurrWindowsBlocks(windowBlockIndeces, 0, windowSide, stepSize, n, i, j);
						currDescriptorsTogether = returnFeaturesOfCurrWindow(windowBlockIndeces, _allKeypoints, _allDescriptors, _currKeypoints, _currDescriptors, c.returnDescriptorSize(), i,j);
						//printWindowFeatureBlocks(_currKeypoints, _currDescriptors, i, j, l, stepSize);
					} else if (i > 0 &&j == 0) { // recall of the support indeces + down shift [1] + update of the support indeces
						windowBlockIndeces = supportIndeces;
						updateCurrWindowsBlocks(windowBlockIndeces, 1, windowSide, stepSize, n, i, j);
						supportIndeces = windowBlockIndeces;
						currDescriptorsTogether = returnFeaturesOfCurrWindow(windowBlockIndeces, _allKeypoints,_allDescriptors, _currKeypoints,_currDescriptors, c.returnDescriptorSize(), i, j);
						//printWindowFeatureBlocks(_currKeypoints, _currDescriptors, i, j, l, stepSize);
					} else { // right shift [0]
						updateCurrWindowsBlocks(windowBlockIndeces, 0, windowSide, stepSize, n, i, j);
						currDescriptorsTogether = returnFeaturesOfCurrWindow(windowBlockIndeces, _allKeypoints, _allDescriptors, _currKeypoints, _currDescriptors, c.returnDescriptorSize(), i, j);
						//printWindowFeatureBlocks(_currKeypoints, _currDescriptors, i, j, l, stepSize);
					}

					fromVecOfVec2Vec(_currKeypoints, currKeypointsForDrawing);
					//cv::drawKeypoints(tmpInputImage, currKeypointsForDrawing, inputImageCopyForDrawing2, cv::Scalar(255,255,0));

					int windowNumber = n * i + j;

					inputImageCopyForDrawing2.copyTo(imageForDrawing);
					cv::Mat currDescriptorsTogetherWithoutZeroRows;

					//printDescriptors("descriptorsBefore.txt", currDescriptorsTogether);

					if (!currDescriptorsTogether.empty()) {
						cv::Mat bagOfCurrWindow = returnBoWOFCurrentWindow(windowBlockIndeces, BoWsOfAllBlocks);

						int prediction = static_cast<int>(svm->predict(bagOfCurrWindow));

						cv::Mat rawPrediction;
						svm->predict(bagOfCurrWindow, rawPrediction, cv::ml::StatModel::Flags::RAW_OUTPUT);
						float dist = rawPrediction.at<float>(0, 0);
						double confidence = (2.0 / (1.0 + std::exp(-std::abs(dist)))) - 1.0;
						//std::cout<<"Prediction ["<<i<<", "<<j<<"] = "<<prediction<<", confidence "<<confidence<<", dist = "<<dist<<std::endl;
						double minConf;
						if (nKeypoints <= 50)
							minConf = 0.00001;
						else
							minConf = 0.00037;
						if (prediction==1 && confidence>=minConf) {
							topLeftCorners[s].push_back(cv::Point2f(j,i));
							cv::Scalar keypointsColor(255, 128, 0);
							cv::drawKeypoints(inputImageCopyForDrawing2, currKeypointsForDrawing, inputImageCopyForDrawing2, keypointsColor);
							cv::Scalar rectanglesColor(0,0,200);
							drawRectangles(false, l, i, j, inputImageCopyForDrawing2, stepSize, windowSide, rectanglesColor);
							cv::namedWindow("slidingWindow", cv::WINDOW_NORMAL);
							cv::resizeWindow("slidingWindow", N, M);
							cv::imshow("slidingWindow", inputImageCopyForDrawing2);

						}else{
							cv::Scalar keypointsColor(255, 255, 0);
							cv::drawKeypoints(inputImageCopyForDrawing2, currKeypointsForDrawing, imageForDrawing, keypointsColor);
							cv::Scalar rectanglesColor(255, 0, 0);
							drawRectangles(false, l, i, j, imageForDrawing, stepSize, windowSide, rectanglesColor);
							cv::namedWindow("slidingWindow", cv::WINDOW_NORMAL);
							cv::resizeWindow("slidingWindow", N, M);
							cv::imshow("slidingWindow", imageForDrawing);
						}
					} else {
						cv::Scalar rectanglesColor(0, 255, 0);
						drawRectangles(false, l, i, j, imageForDrawing, stepSize, windowSide, rectanglesColor);
						cv::namedWindow("slidingWindow", cv::WINDOW_NORMAL);
						cv::resizeWindow("slidingWindow", N, M);
						cv::imshow("slidingWindow", imageForDrawing);
					}
					int key = cv::waitKey(wait);
					saveWindow(key, wait, i, j, windowSide, stepSize, inputImage);
				}
			}
			std::cout<<"Sliding window procedure completed."<<std::endl;
//			std::cout<<"Image level "<<s<<std::endl;
//			for (int i = 0; i < topLeftCorners[s].size(); i++) {
//				std::cout<<"TopLeftCorner["<<i<<"] = ["<<topLeftCorners[s][i].x<<", "<<topLeftCorners[s][i].y<<"]"<<std::endl;
//			}

			std::vector<std::vector<int>> currFinalBoxesNotMergedBoxes;
			currFinalBoxesNotMergedBoxes = returnIntermediateBoxes(topLeftCorners[s],l,0.2);

			for (int p = 0; p < currFinalBoxesNotMergedBoxes.size(); p++) {
				int i = currFinalBoxesNotMergedBoxes[p][0];
				int j = currFinalBoxesNotMergedBoxes[p][1];
				int w = currFinalBoxesNotMergedBoxes[p][2];
				int h = currFinalBoxesNotMergedBoxes[p][3];
				// drawing not refined bounding box
				//cv::rectangle(inputImageCopyForDrawing2,cv::Point(j*stepSize,i*stepSize),cv::Point((j+w)*stepSize,(i+h)*stepSize),cv::Scalar(0,0,255),2);
				double xMin;
				double yMin;
				double wNew;
				double hNew;
				refineBox(i,j,w,h, xMin, yMin, wNew, hNew, n, stepSize,_allKeypoints);

				cv::Rect roi(xMin,yMin, wNew, hNew);
				cv::Mat candidateBoat;
				inputImage(roi).copyTo(candidateBoat);
				std::vector<double> v{xMin, yMin, wNew, hNew, static_cast<double>(s)};
				finalBoxesOfAllLayers.push_back(v);
				cv::Scalar rectanglesColor;
				if (s==0)
					rectanglesColor = cv::Scalar(0, 153, 255);
				else if (s==1)
					rectanglesColor = cv::Scalar(255, 0, 204);
				else if (s==2)
					rectanglesColor = cv::Scalar(204, 204, 51);
				cv::rectangle(inputImageCopyForDrawing2,cv::Point(xMin,yMin),cv::Point(xMin+wNew,yMin+hNew),rectanglesColor,2);
			}
			//cv::imwrite("outputLevel"+std::to_string(s+1)+".jpg", inputImageCopyForDrawing2);
			cv::namedWindow("slidingWindow", cv::WINDOW_NORMAL);
			cv::resizeWindow("slidingWindow", N, M);
			cv::imshow("slidingWindow", inputImageCopyForDrawing2);
			std::cout<<"To continue press any key."<<std::endl;
			cv::waitKey(0);


			// End of the research in the current pyramid level
			std::cout<<"End research in the level "<<s+1<<" of image pyramid -> preparation of the next level image.\n"<<std::endl;
			cv::pyrDown(inputImage, inputImage);

			N = inputImage.cols;
			M = inputImage.rows;

			cv::resize(inputImage, inputImage, cv::Size(N, M), cv::INTER_LINEAR);
		}

		std::cout<<"---------------   End of the search.   --------------- "<<std::endl;

		std::cout<<"Upsampling all the boxes..."<<std::endl;
		for (int i = 0; i < finalBoxesOfAllLayers.size(); i++) {
			int s = finalBoxesOfAllLayers[i][4];
			cv::Scalar color;
			if (s==0)
				color = cv::Scalar(0, 153, 255);
			else if (s==1)
				color = cv::Scalar(255, 0, 204);
			else if (s==2)
				color = cv::Scalar(204, 204, 51);
			resizeBox(finalBoxesOfAllLayers[i][0],finalBoxesOfAllLayers[i][1],finalBoxesOfAllLayers[i][2],finalBoxesOfAllLayers[i][3],std::pow(2,s));
			double xMin = finalBoxesOfAllLayers[i][0];
			double yMin= finalBoxesOfAllLayers[i][1];
			double w = finalBoxesOfAllLayers[i][2];
			double h = finalBoxesOfAllLayers[i][3];
			cv::rectangle(inputImageAllScaleBoxes,cv::Point(xMin, yMin),cv::Point(xMin + w, yMin+h), color, 2);
		}
		std::cout<<"Upsampling complete."<<std::endl;
		std::cout<<"Showing the final boxes of each level: orange -> level 1, purple -> level 2, turquoise -> level 3."<<std::endl;
		cv::namedWindow("finalResult", cv::WINDOW_NORMAL);
		N = inputImageAllScaleBoxes.cols;
		M = inputImageAllScaleBoxes.rows;
		cv::resizeWindow("finalResult", N, M);
		cv::imshow("finalResult", inputImageAllScaleBoxes);
		cv::imwrite("finalResult0.jpg", inputImageAllScaleBoxes);
		std::cout<<"To continue press any key."<<std::endl;
		cv::waitKey(0);

		finalBoxesNotMergedLayer0 = extractFinalSameLevelNotMergedBoxes(finalBoxesOfAllLayers, 0);
		finalBoxesNotMergedLayer1 = extractFinalSameLevelNotMergedBoxes(finalBoxesOfAllLayers, 1);
		finalBoxesNotMergedLayer2 = extractFinalSameLevelNotMergedBoxes(finalBoxesOfAllLayers, 2);

		std::vector<std::vector<double>> finalBoxesLevel0 = returnFinalBoxes(finalBoxesNotMergedLayer0, 0.20);
		for (int i = 0; i < finalBoxesLevel0.size(); i++) {
			finalBoxesLevel0[i].push_back(0);
		}
		std::vector<std::vector<double>> finalBoxesLevel1 = returnFinalBoxes(finalBoxesNotMergedLayer1, 0.20);
		for (int i = 0; i < finalBoxesLevel1.size(); i++) {
			finalBoxesLevel1[i].push_back(1);
		}
		std::vector<std::vector<double>> finalBoxesLevel2 = returnFinalBoxes(finalBoxesNotMergedLayer2, 0.20);
		for (int i = 0; i < finalBoxesLevel2.size(); i++) {
			finalBoxesLevel2[i].push_back(2);
		}

		std::vector<std::vector<double>> finalMergedBoxesAllLevels;
		finalMergedBoxesAllLevels.insert(finalMergedBoxesAllLevels.end(), finalBoxesLevel0.begin(), finalBoxesLevel0.end());
		finalMergedBoxesAllLevels.insert(finalMergedBoxesAllLevels.end(), finalBoxesLevel1.begin(), finalBoxesLevel1.end());
		finalMergedBoxesAllLevels.insert(finalMergedBoxesAllLevels.end(), finalBoxesLevel2.begin(), finalBoxesLevel2.end());

		for (int i = 0; i < finalMergedBoxesAllLevels.size(); i++) {
			int s = finalMergedBoxesAllLevels[i][4];
			cv::Scalar color;
			if (s==0)
				color = cv::Scalar(0, 153, 255);
			else if (s==1)
				color = cv::Scalar(255, 0, 204);
			else if (s==2)
				color = cv::Scalar(204, 204, 51);
			double xMin = finalMergedBoxesAllLevels[i][0];
			double yMin= finalMergedBoxesAllLevels[i][1];
			double w = finalMergedBoxesAllLevels[i][2];
			double h = finalMergedBoxesAllLevels[i][3];
			cv::rectangle(inputImageTemporaryResult,cv::Point(xMin, yMin),cv::Point(xMin + w, yMin+h), color, 2);
		}

		std::cout<<"Showing the result of the mergers of the boxes of a same level with overlap >=20% for at least one of the boxes involved."<<std::endl;
		cv::namedWindow("finalResult", cv::WINDOW_NORMAL);
		cv::resizeWindow("finalResult", N, M);
		cv::imshow("finalResult", inputImageTemporaryResult);
		cv::imwrite("finalResult1.jpg", inputImageTemporaryResult);
		std::cout<<"To continue press any key."<<std::endl;
		cv::waitKey(0);

		double overlappingRatio = 1;
		pruneBoxes(finalMergedBoxesAllLevels, overlappingRatio);

		std::vector<std::vector<double>> finalBoxes = returnFinalBoxes(finalMergedBoxesAllLevels, 0.75);

		std::cout<<"Drawing the GT boxes in green..."<<std::endl;
		//std::cout<<"Number of GT boxes: "<< GTCoords.size()<<std::endl;
		for (int p = 0; p < GTCoords.size(); p++) {
			double xMin = GTCoords[p][0];
			double yMin = GTCoords[p][1];
			double wNew = GTCoords[p][2];
			double hNew = GTCoords[p][3];
			cv::rectangle(inputImageFinalResult,cv::Point(xMin,yMin),cv::Point(xMin+wNew,yMin+hNew),cv::Scalar(0,255,0),2);
		}

		std::cout<<"Drawing the final detected boxes in blue..."<<std::endl;
		for (int p = 0; p < finalBoxes.size(); p++) {
			double xMin = finalBoxes[p][0];
			double yMin = finalBoxes[p][1];
			double wNew = finalBoxes[p][2];
			double hNew = finalBoxes[p][3];
			cv::rectangle(inputImageFinalResult,cv::Point(xMin,yMin),cv::Point(xMin+wNew,yMin+hNew),cv::Scalar(255,0,0),2);
		}

		cv::namedWindow("finalResult", cv::WINDOW_NORMAL);
		cv::resizeWindow("finalResult", N, M);
		cv::imshow("finalResult", inputImageFinalResult);
		cv::imwrite("finalResult2.jpg", inputImageFinalResult);
		std::cout<<"To continue press any key."<<std::endl;
		cv::waitKey(0);

		std::cout<<"Drawing the union between the detected boxes and the GT boxes in red..."<<std::endl;
		std::vector<std::vector<std::vector<double>>> unionPoligon = returnUnionPoligon(finalBoxes,GTCoords);

		for (int i = 0; i < unionPoligon.size(); i++) {
			for (int j = 0; j < unionPoligon[i].size()-1; j++) {
				double x1 = unionPoligon[i][j][0];
				double y1 = unionPoligon[i][j][1];
				double x2 = unionPoligon[i][j+1][0];
				double y2 = unionPoligon[i][j+1][1];
				cv::line(inputImageFinalResult,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,0,255),2);
			}
		}

		double unionArea = returnArea(unionPoligon);

		std::cout<<"Drawing the intersection between the detected boxes and the GT boxes in yellow..."<<std::endl;
		std::vector<std::vector<std::vector<double>>> intersectionPoligon = returnIntersectionPoligon(finalBoxes, GTCoords);
		for (int i = 0; i < intersectionPoligon .size(); i++) {
			for (int j = 0; j < intersectionPoligon [i].size()-1; j++) {
				double x1 = intersectionPoligon [i][j][0];
				double y1 = intersectionPoligon [i][j][1];
				double x2 = intersectionPoligon [i][j+1][0];
				double y2 = intersectionPoligon [i][j+1][1];
				cv::line(inputImageFinalResult,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,255,255),2);
			}
		}

		double intersectionArea = returnArea(intersectionPoligon);

		if (!GTCoords.empty()) {
			double IoU = intersectionArea / unionArea;
			std::cout << "IoU = " << IoU << std::endl;
		}else{
			std::cout<<"There are no GT boxes for the current image => IoU not computed."<<std::endl;
		}

		cv::namedWindow("finalResult", cv::WINDOW_NORMAL);
		cv::resizeWindow("finalResult", N, M);
		cv::imshow("finalResult", inputImageFinalResult);
		cv::imwrite("finalResult3.jpg", inputImageFinalResult);
		std::cout<<"To exit press any key."<<std::endl;
		cv::waitKey(0);
	}
	return 0;
}
// END MAIN

// ----------   FUNCTION DEFINITIONS  ----------

std::string selector(int caseParam){
	bool stop = true;
	cv::String optionSelected;
	while(stop){
		std::cin >> optionSelected;
		if (caseParam==0) {
			if (optionSelected.compare("0") != 0 && optionSelected.compare("1") != 0) {
				std::cout << "Error: type \"0\" or \"1\": ";
				std::cin.clear();
				std::cin.ignore(10000, '\n');
			} else {
				stop = false;
			}
		}else if (caseParam==1) {
			if (optionSelected.compare("y") != 0 && optionSelected.compare("n") != 0) {
				std::cout << "Error: type \"y\" or \"n\": ";
				std::cin.clear();
				std::cin.ignore(10000, '\n');
			} else {
				stop = false;
			}
		}
	}
	return optionSelected;
}

// ----------   FUNCTION DEFINITIONS: (MOSTLY) OFFLINE PHASE   ----------
void loadImages(std::vector<cv::Mat>& images, std::string s){
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

cv::Mat computeFeatureDescriptors(std::vector<cv::Mat>& examples, std::vector<cv::Mat>& descriptors){
	std::cout<<"Computation of the features descriptors..."<<std::endl;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0,3,0.04,10,1.6);
	std::vector<std::vector<cv::KeyPoint>> keypoints(examples.size());
	std::cout<<"Example ";
	for (int i = 0; i < descriptors.size(); i++) {
		std::string s = std::to_string(i+1) + "/" + std::to_string(descriptors.size());
		std::cout<<s;
		std::cout << std::string(s.length(),'\b');
		detector->detectAndCompute(examples[i],cv::Mat(), keypoints[i], descriptors[i], false);
		//cv::drawKeypoints(examples[i],keypoints[i],examples[i],cv::Scalar(0,128,255));
		//cv::imwrite("checkPositiveExamples\\img"+std::to_string(i)+".jpg",examples[i]);
		//std::cout<<"keypoints["<<i<<"].size() = "<<keypoints[i].size()<<std::endl;
		//std::cout<<"descriptors["<<i<<"].size() = ["<<descriptors[i].rows<<", "<<descriptors[i].cols<<"]"<<std::endl;
	}
	std::cout<<"\n";
	cv::Mat allDescriptors;
	cv::vconcat(descriptors, allDescriptors);
	std::cout<<"Computation completed: allDescriptors.size() = ["<<allDescriptors.rows<<", "<<allDescriptors.cols<<"]"<<std::endl;
	return allDescriptors;
}

void loadInputImage(cv::Mat& inputImage){
	cv::String img;
	bool stop = false;
	while(!stop){
		std::cout<<"Please insert the path of the input image:"<<std::endl;
		std::cin >> img;
		inputImage = cv::imread(img);
		if (inputImage.empty()){
			std::cout<<"Error: the file \""<<img<<"\" cannot be found or is not an image file."<<std::endl;
		} else
			stop = true;
	}
	std::cout << "Input image loaded."<<std::endl;
}

bool checkPath(const std::string &s){
	struct stat buffer{};
	return (stat (s.c_str(), &buffer) == 0);
}

void readGroundTruthCoordsFromTxT(std::vector<cv::Mat> codebookImages, std::vector<std::vector<int>>& gtCoordinates){
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
		if (txtFile.eof()){
			//std::cout<<"EOF"<<std::endl;
			txtFile.close();
			break;
		}
	}
}

void extractBoatExamples(std::vector<cv::Mat> codebookImages, std::vector<cv::Mat>& boatExamples, std::vector<std::vector<int>>& gtCoordinates){
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
			boatExamples.push_back(currBoatExample);
		}
	}
	std::cout<<"Extraction complete: extracted " <<boatExamples.size()<<" examples (sub-images)."<<std::endl;
}

// ----------   FUNCTION DEFINITIONS: ONLINE PHASE   ----------
int getNearestDivisibleInt(int stepSize, int length){
	if (length % stepSize==0)
		return length;
	std::vector<int> leftRightInt(2);
	int step;
	int lengthCopy = length;
	for (int i = 0; i < 2; i++) {
		length = lengthCopy;
		if (i==0)
			step = -1;
		else
			step = 1;
		while (length % stepSize != 0){
			length = length + step;
			if (length % step==0)
				leftRightInt[i] = length;
		}
	}
	std::cout<<"leftInteger = "<<leftRightInt[0]<<std::endl;
	std::cout<<"rightInteger = "<<leftRightInt[1]<<std::endl;
	if (abs(leftRightInt[0]-length)<=abs(leftRightInt[1]-length))
		return leftRightInt[0];
	else
		return leftRightInt[1];
}

void computeFeatureBlocks(cv::Mat inputImage, int windowSide, int stepSize, std::vector<std::vector<cv::KeyPoint>>& allKeypoints, std::vector<cv::Mat>& allDescriptors){
	std::cout<<"Performing the feature assignment to the blocks..."<<std::endl;
	cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0,3,0.04,10,1.6);
	std::vector<cv::KeyPoint> tmpKeypoints;
	cv::Mat tmpDescriptors;
	detector->detectAndCompute(inputImage, cv::Mat(), tmpKeypoints, tmpDescriptors,false);
	int n = inputImage.cols/stepSize;
	int m = inputImage.rows/stepSize;
	int currXMin;
	int currXMax;
	int currYMin;
	int currYMax;
	std::cout<<"Keypoint ";
	int i;
	int j;
	for (int h = 0; h < tmpKeypoints.size(); h++) {
		i = static_cast<int>(tmpKeypoints[h].pt.y/stepSize);
		j = static_cast<int>(tmpKeypoints[h].pt.x/stepSize);
		allKeypoints[n*i+j].push_back(tmpKeypoints[h]);
		cv::Rect row(0,h,tmpDescriptors.cols,1);
		allDescriptors[n*i+j].push_back(tmpDescriptors(row));
		std::string s = std::to_string(h+1) + "/" + std::to_string(tmpKeypoints.size()) +" assigned to block "+std::to_string(n*i+j)+"/"+std::to_string(allKeypoints.size());
		std::cout<< s;
		std::cout << std::string(s.length(),'\b');
	}
	//printImageFeatureBlocks(allKeypoints, allDescriptors, n, stepSize);
	std::cout<<"\nFeature assigment completed."<<std::endl;
}

void initializeWindowsBlocks(std::list<int>& windowBlockIndeces, int windowSide, int stepSize, int n, int i, int j){
	//std::cout<<"Initializing the window's blocks..."<<std::endl;
	int l = windowSide/stepSize;
	for (int ii = 0; ii < l; ii++) {
		for (int jj = 0; jj < l; jj++) {
			windowBlockIndeces.push_back(i*n+j+jj*n+ii);
		}
	}
	//std::cout<<"Initialization complete."<<std::endl;
}

void updateCurrWindowsBlocks(std::list<int>& windowBlockIndeces, int shift, int windowSide, int stepSize, int n, int i, int j){
	//std::cout<<"Starting the updating of the window's blocks..."<<std::endl;
	int l = windowSide/stepSize;
	if (shift==0){ // Right shift
		for (int ii = 0; ii < l; ii++) {
			windowBlockIndeces.pop_front();
		}
		if (l>1){
			int p = l*(l-2);
			std::vector<int> v(l);
			for (int ii = 0; ii < l; ii++) {
				std::list<int>::iterator it = std::next(windowBlockIndeces.begin(), p+ii);
				v[ii] = *it+1;
			}
			for (int ii = 0; ii < l; ii++) {
				windowBlockIndeces.push_back(v[ii]);
			}
		} else {
			initializeWindowsBlocks(windowBlockIndeces, windowSide, stepSize, n, i, j);
		}
	}else if(shift == 1){ // Down shift: happens only with column index j of the top left block equal to j=0
		for (int i = 0; i < l; i++) {
			std::list<int>::iterator it = std::next(windowBlockIndeces.begin(), (l-1)*i);
			windowBlockIndeces.erase(it);
		}
		if (l>1){
			for (int i = 0; i < l; i++) {
				std::list<int>::iterator it1 = std::next(windowBlockIndeces.begin(), (l-1)*(i+1)+i); //For pointing the position of insertion
				std::list<int>::iterator it2 = std::next(windowBlockIndeces.begin(), (l-1)*(i+1)+i-1); //For pointing the position before for ispection
				windowBlockIndeces.insert(it1,*it2+n);
			}
		} else {
			initializeWindowsBlocks(windowBlockIndeces, windowSide, stepSize, n, i, j);
		}
	}
	//std::cout<<"Updating comple."<<std::endl;
}

// This function returns the descriptors of the current window in one single matrix and writes the current
// keypoints and the current descriptors in the inputs std::vector<cv::KeyPoint> currKeypoints and
// cv::Mat currDescriptors
cv::Mat returnFeaturesOfCurrWindow(std::list<int> windowBlockIndeces,  std::vector<std::vector<cv::KeyPoint>> allKeypoints, std::vector<cv::Mat> allDescriptors,
								   std::vector<std::vector<cv::KeyPoint>>& currKeypoints, std::vector<cv::Mat>& currDescriptors, int descriptorSize, int r, int c){
	//std::cout<<"---------- window ["<<r<<", "<<c<<"] ----------"<<std::endl;
	int i=0;
	int totCurrKeypoints = 0;
	int totCurrDescriptors = 0;
	for (std::list<int>::iterator it = windowBlockIndeces.begin(); it != windowBlockIndeces.end(); it++) {
		//std::cout<<"IMAGE BLOCK "<<*it<<":"<<std::endl;
		if (allKeypoints[*it].size()>0){
			currKeypoints[i] = allKeypoints[*it];
			//std::cout<<"currKeypoints[window block "<<i<<"].size() = "<<currKeypoints[i].size()<<std::endl;
			totCurrKeypoints += currKeypoints[i].size();
			allDescriptors[*it].copyTo(currDescriptors[i]);
			//std::cout<<"currDescriptor[window block "<<i<<"].size() = ["<<currDescriptors[i].rows<<", "<<currDescriptors[i].cols<<"];"<<std::endl;
			totCurrDescriptors += currDescriptors[i].rows;
		} else{ // A trick in order to concatenate the matrix even where some are empty
			//std::cout<<"allKeypoints[image block "<<*it<<"] is empty"<<std::endl;
			currKeypoints[i].clear();
			cv::Mat tmp(0,descriptorSize, CV_32F, cv::Scalar::all(0));
			tmp.copyTo(currDescriptors[i]);
		}
		i++;
	}

	cv::Mat allDescriptorsInOneMatrix;
	//std::cout<<"Total current keypoints = "<<totCurrKeypoints<<std::endl;
	//std::cout<<"non-zero descriptors/total descriptors matrices= "<<totCurrDescriptors<<"/"<<currDescriptors.size()<<std::endl;
	//std::cout<<"currDescriptors.size() = "<<currDescriptors.size()<<std::endl;
	personalVConcat(currDescriptors, allDescriptorsInOneMatrix); // automatically ignores the 0x128 matrices currDescriptors[i]
	//std::cout<<"allDescriptorsInOneMatrix.size() = ["<<allDescriptorsInOneMatrix.rows<<", "<<allDescriptorsInOneMatrix.cols<<"];"<<std::endl;

	return allDescriptorsInOneMatrix;
}


void fromVecOfVec2Vec(std::vector<std::vector<cv::KeyPoint>> vecOfVec, std::vector<cv::KeyPoint>& vec){
	int k = 0;
	vec.clear();
	for (int i = 0; i < vecOfVec.size(); i++) {
		std::vector<cv::KeyPoint> currVec = vecOfVec[i];
		for (int j = 0; j < currVec.size(); j++) {
			vec.push_back(currVec[j]);
			k++;
		}
	}
	//std::cout<<"fromVecOfVec2Vec: resulting vector size = "<<vec.size()<<std::endl;
}

void personalVConcat(std::vector<cv::Mat> matrices, cv::Mat& globMatrix){
	cv::Mat tmpGlobMatrix;
	for (int i = 0; i < matrices.size(); i++) {
		cv::Mat currMatrix = matrices[i];
		if (!currMatrix.empty()){
			tmpGlobMatrix.create(tmpGlobMatrix.rows+currMatrix.rows, currMatrix.cols, CV_32F);
			cv::vconcat(tmpGlobMatrix, currMatrix);
		}
	}
	tmpGlobMatrix.copyTo(globMatrix);
}

void takeInputsWindowSideAndStepSize(int& windowSide, int& stepSize){
	bool stop = false;
	double ws;
	while(!stop){
		std::cout << "Insert the window size: ";
		std::cin >> ws;
		if (std::cin.fail()) {
			std::cout << "Something went wrong: check the window side inserted." <<std::endl;
			std::cin.clear();
			std::cin.ignore(10000, '\n');
		}else{
			windowSide = static_cast<int>(ws);
			std::cout<<"Aquired windowSize = "<<windowSide<<";"<<std::endl;
			stop = true;
		}
		std::cout<<"\n";
	}
	stop = false;
	double ss;
	while(!stop) {
		std::cout << "Insert the step size (it must divide the window side): ";
		std::cin >> ss;
		if (std::cin.fail()) {
			std::cout << "Something went wrong: check the step size inserted." << std::endl;
			std::cin.clear();
			std::cin.ignore(10000, '\n');
		} else if (windowSide % static_cast<int>(ss) != 0) {
			std::cout << "Error: the window side is not divisible by " << ss
					  << ". Please insert a step size which divides the window side:" << std::endl;
		} else {
			stepSize = static_cast<int>(ss);
			std::cout << "Aquired stepSize = " << stepSize << ";" << std::endl;
			stop = true;
		}
		std::cout << "\n";
	}
}

void drawRectangles(bool drawAll, int l, int i, int j, cv::Mat& inputImageCopyForDrawing, int stepSize, int windowSide, cv::Scalar color){
	if (drawAll){
		std::vector<cv::Point2f> topLeftCorners(l*l);
		std::vector<cv::Point2f> bottomRightCorners(l*l);
		int q = 0;
		for (int k = 0; k < l; k++) {
			for (int p = 0;  p < l; p++) {
				topLeftCorners[q] = cv::Point2f(j*stepSize+p*stepSize, i*stepSize+k*stepSize);
				bottomRightCorners[q] = cv::Point2f(j*stepSize+p*stepSize + stepSize, i*stepSize+k*stepSize + stepSize);
				q++;
			}
		}
		for (int k = 0; k < l*l; k++) {
			cv::rectangle(inputImageCopyForDrawing, topLeftCorners[k], bottomRightCorners[k], color, 2);
		}
	}else{
		cv::Point2f topLeftCorner(j*stepSize, i*stepSize);
		cv::Point2f bottomRightCorner(j*stepSize+windowSide, i*stepSize+windowSide);
		cv::rectangle(inputImageCopyForDrawing, topLeftCorner, bottomRightCorner, color, 2);
	}

//	cv::Point2f topLeftCorner(j*stepSize, i*stepSize);
//	cv::Point2f bottomRightCorner(j*stepSize+windowSide, i*stepSize+windowSide);
}

cv::Mat returnBoWsOfTheExamples(std::vector<cv::Mat> descriptors, Codebook c, std::string s){
	std::cout<<"Computing the BoWs of the "<<s<<": please be patient it could take few minutes..."<<std::endl;
	cv::Mat outputBoWs;
	//std::cout<<"outputBoWs.size() = ["<<outputBoWs.rows<<", "<<outputBoWs.cols<<"]"<<std::endl;
	std::cout<<"Example ";
	double ratioThreshold = 0.95;
	double distanceThreshold = 1000;
	for (int i = 0; i < descriptors.size(); i++) {
		std::string s = std::to_string(i + 1) + "/" + std::to_string(descriptors.size());
		std::cout << s;
		std::cout << std::string(s.length(), '\b');
		cv::Mat ithBoW = returnBoWFromDescriptors(descriptors[i], c, 0,ratioThreshold, distanceThreshold);
		//std::cout<<"  ithBoW.size() = ["<<ithBoW.rows<<", "<<ithBoW.cols<<"]"<<std::endl;
		outputBoWs.push_back(ithBoW);
		//std::cout<<"  outputBoWs.size() = ["<<outputBoWs.rows<<", "<<outputBoWs.cols<<"]"<<std::endl;

	}
	//std::cout<<"outputBoWs.size() = ["<<outputBoWs.rows<<", "<<outputBoWs.cols<<"]"<<std::endl;
	return outputBoWs;
}

cv::Mat returnBoWFromDescriptors(cv::Mat descriptors, Codebook c, int windowNumber, double ratioThreshold, double distanceThreshold){ //phase = 0 => offline, phase = 1 =>online
	cv::Mat codebook = c.returnCodebook();
	int descriptorSize = c.returnDescriptorSize();
	cv::Mat outputBoW(1, c.returnCodebookSize(), CV_32F, cv::Scalar::all(0));

	if (outputBoW.at<float>(0,c.returnCodebookSize()-1) != 0) {
		outputBoW.at<float>(0, c.returnCodebookSize() - 1) = 0;
	}

	std::vector<int> firstNeighIdx(descriptors.rows,-1);
	std::vector<int> secondNeighIdx(descriptors.rows);
	std::vector<double> firstMinDist(descriptors.rows);
	std::vector<double> secondMinDist(descriptors.rows);

	for (int i = 0; i < descriptors.rows; i++) {
		// The 1st thing to do is to
		cv::Rect windowRow(0, i, descriptorSize, 1);
		int k = 0;
		int assigment = 0;
		std::vector<double> distVec(2);
		std::vector<int> idxVec(2);
		while (k < codebook.rows && assigment < 2) {
			cv::Rect row(0, k, descriptorSize, 1);
			double dist = cv::norm(descriptors(windowRow), codebook(row), cv::NORM_L2);
			if (!cvIsNaN(dist)) {
				distVec[assigment] = dist;
				idxVec[assigment] = k;
				assigment++;
			}
			k++;
		}
		if (assigment == 2) {
			double dist1 = distVec[0];
			double dist2 = distVec[1];
			if (dist1 <= dist2) {
				firstMinDist[i] = dist1;
				firstNeighIdx[i] = 0;
				secondMinDist[i] = dist2;
				secondNeighIdx[i] = 1;
			} else {
				firstMinDist[i] = dist2;
				firstNeighIdx[i] = 1;
				secondMinDist[i] = dist1;
				secondNeighIdx[i] = 0;
			}
			for (int j = k; j < codebook.rows ; j++) {
				cv::Rect codebookRow(0, j, descriptorSize, 1);
				double currDist = cv::norm(descriptors(windowRow), codebook(codebookRow), cv::NORM_L2);
				if (!cvIsNaN(currDist) && currDist<firstMinDist[i]){
					secondMinDist[i] = firstMinDist[i];
					secondNeighIdx[i] = firstNeighIdx[i];
					firstMinDist[i] = currDist;
					firstNeighIdx[i] = j;
				} else if (!cvIsNaN(currDist) && currDist<secondMinDist[i]){
					secondMinDist[i] = currDist;
					secondNeighIdx[i] = j;
				}
			}
		}

		//std::cout<<"Descriptor "<<i<<": the 1st nearest codeword is the codeword number "<<firstNeighIdx[i]<<" with distance = "<<firstMinDist[i]<<std::endl;
		//std::cout<<"Descriptor "<<i<<": the 2nd nearest codeword is the codeword number "<<secondNeighIdx[i]<<" with distance = "<<secondMinDist[i]<<std::endl;
	}

	//std::cout<<"--------------------       Window number "<<windowNumber<<"       --------------------"<<std::endl;
	//std::cout<<"Total features detected = "<<firstNeighIdx.size()<<std::endl;

	for (int i = 0; i < firstNeighIdx.size(); i++) {
		if (firstNeighIdx[i]!=-1 && secondNeighIdx[i]!=-1){
			if((firstMinDist[i]/secondMinDist[i])<ratioThreshold && firstMinDist[i]<distanceThreshold) //
			outputBoW.at<float>(0,firstNeighIdx[i]) = outputBoW.at<float>(0,firstNeighIdx[i]) + 1;
		}

	}

//	std::cout<<"--- prima penultimo for returnBag..."<<std::endl;
//	for (int i = 0; i < bagOfTheCurrWindow.cols; i++) {
//		if (bagOfTheCurrWindow.at<float>(0,i)>=1)
//			std::cout<<"[codeword number, total matches in the window] = ["<<i<<", "<<bagOfTheCurrWindow.at<float>(0,i)<<"]"<<std::endl;
//	}
	return outputBoW;
}

double returnFractionOfOverlappedArea(int i1, int j1, int i2, int j2, int l){
	double jSide = j1 <= j2 ? j1+l-j2 : j2+l-j1;
	double iSide = i1 <= i2 ? i1+l-i2 : i2+l-i1;
	double area;
	if(jSide>0 && iSide>0)
		area = (jSide * iSide)/(l*l);
	else
		area = 0;
	return area;
}

double returnAbsoluteOverlappedArea(double xMin1, double yMin1, double w1, double h1, double xMin2, double yMin2, double w2, double h2){
	double xMax1 = xMin1 + w1;
	double xMax2 = xMin2 + w2;

	double yMax1 = yMin1 + h1;
	double yMax2 = yMin2 + h2;

	double xSideStart = xMin1 <=  xMin2? xMin2 : xMin1;
	double xSideEnd = xMax1 <=  xMax2? xMax1 : xMax2;
	double xSide = xSideEnd - xSideStart;
	//std::cout<<"  intersection xStart = "<<xSideStart<<std::endl;
	//std::cout<<"  intersection xEnd = "<<xSideEnd<<std::endl;
	//std::cout<<"  intersection xSide = "<<xSide<<std::endl;

	double ySideStart = yMin1 <=  yMin2? yMin2 : yMin1;
	double ySideEnd = yMax1 <=  yMax2? yMax1 : yMax2;
	double ySide = ySideEnd - ySideStart;
	//std::cout<<"  intersection yStart = "<<ySideStart<<std::endl;
	//std::cout<<"  intersection yEnd = "<<ySideEnd<<std::endl;
	//std::cout<<"  intersection ySide = "<<ySide<<std::endl;

	double area = 0;
	if(xSide>0 && ySide>0) {
		area = xSide*ySide;
	}
	//std::cout<<"  intersection area = "<<area<<std::endl;
	return area;
}

std::vector<std::vector<int>> returnIntermediateBoxes(std::vector<cv::Point2f> currImgTopLeftCorners, int l, double minOverlap){
	std::vector<std::vector<int>> overlappingWindowsTopLeftCorners(currImgTopLeftCorners.size()); //For each window compute the windows with overlap>=minOverlap
	for (int p = 0; p < currImgTopLeftCorners.size(); p++) {
		int i1 = static_cast<int>(currImgTopLeftCorners[p].y);
		int j1 = static_cast<int>(currImgTopLeftCorners[p].x);
		//std::cout<<"Windows with overlap > "<<minOverlap<<" with window "<<p<<":"<<std::endl;
		for (int h = 0; h < currImgTopLeftCorners.size(); h++) {
			if (h!=p){
				int i2 = static_cast<int>(currImgTopLeftCorners[h].y);
				int j2 = static_cast<int>(currImgTopLeftCorners[h].x);
				double overlap = returnFractionOfOverlappedArea(i1, j1, i2, j2, l);
				if (overlap>minOverlap){
					overlappingWindowsTopLeftCorners[p].push_back(h);
					//std::cout<<"   window "<<h<<" -> overlap = "<<overlap<<std::endl;
				}
			}
		}
	}

	//std::cout<<"Defining a graph object..."<<std::endl;
	Graph g(currImgTopLeftCorners.size(), overlappingWindowsTopLeftCorners);
	int maxConnComp = currImgTopLeftCorners.size();
	std::vector<std::vector<int>> connComp(maxConnComp);
	//std::cout<<"Launching the getConnectedComponents() function..."<<std::endl;
	int actualNumOfConnComp = 0;
	g.getConnectedComponents(connComp, actualNumOfConnComp);
	//g.printConnComp(connComp, actualNumOfConnComp);

	std::vector<int> iMin(actualNumOfConnComp);
	std::vector<int> iMax(actualNumOfConnComp);
	std::vector<int> jMin(actualNumOfConnComp);
	std::vector<int> jMax(actualNumOfConnComp);

	for (int p = 0; p < actualNumOfConnComp; p++) {
		//std::cout<<"connComp["<<p<<"].size() = "<<connComp[p].size()<<std::endl;
		for (int h = 0; h < connComp[p].size(); h++) {
			//std::cout<<"    h = "<<h<<std::endl;
			if (h==0){
				jMin[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].x);
				jMax[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].x) + l;
				iMin[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].y);
				iMax[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].y) + l;
			}else{
				if (currImgTopLeftCorners[connComp[p][h]].x < jMin[p])
					jMin[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].x);
				if (currImgTopLeftCorners[connComp[p][h]].x +l > jMax[p])
					jMax[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].x)+l;
				if (currImgTopLeftCorners[connComp[p][h]].y < iMin[p])
					iMin[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].y);
				if (currImgTopLeftCorners[connComp[p][h]].y +l> iMax[p])
					iMax[p] = static_cast<int>(currImgTopLeftCorners[connComp[p][h]].y)+l;
			}
		}
	}

	std::vector<std::vector<int>> intermediateBoxes(actualNumOfConnComp);
	for (int p = 0; p < actualNumOfConnComp; p++) {
		int w = jMax[p] - jMin[p];
		int h = iMax[p] - iMin[p];
		intermediateBoxes[p].push_back(iMin[p]);
		intermediateBoxes[p].push_back(jMin[p]);
		intermediateBoxes[p].push_back(w);
		intermediateBoxes[p].push_back(h);
	}
	return intermediateBoxes;
}

std::vector<std::vector<double>> returnFinalBoxes(std::vector<std::vector<double>> boxes, double minOverlap){
	std::vector<std::vector<int>> overlappingWindows(boxes.size()); //For each window compute the windows with overlap>=minOverlap
	//std::cout<<"Final number of boxes = "<<boxes.size()<<std::endl;
	//std::cout<<"Searching connected components... "<<std::endl;
	for (int p = 0; p < boxes.size(); p++) {
		double xMin1 = boxes[p][0];
		double yMin1 = boxes[p][1];
		double w1 = boxes[p][2];
		double h1 = boxes[p][3];
		double area1 = w1*h1;
//		std::cout<<"p = "<<p<<std::endl;
//		std::cout<<"xMin1 = "<<xMin1<<std::endl;
//		std::cout<<"xMax1 = "<<yMin1<<std::endl;
//		std::cout<<"w1 = "<<w1<<std::endl;
//		std::cout<<"h1 = "<<h1<<std::endl;
//		std::cout<<"area1 = "<<area1<<std::endl;

//		std::cout<<"Windows with overlap > "<<minOverlap<<" with window "<<p<<":"<<std::endl;
		for (int h = 0; h < boxes.size(); h++) {
			if (h!=p){
//				std::cout<<"  h = "<<h<<std::endl;
				double xMin2 = boxes[h][0];
				double yMin2 = boxes[h][1];
				double w2 = boxes[h][2];
				double h2 = boxes[h][3];
				double area2 = w2*h2;
//				std::cout<<"  xMin2 = "<<xMin2<<std::endl;
//				std::cout<<"  xMax2 = "<<yMin2<<std::endl;
//				std::cout<<"  w2 = "<<w2<<std::endl;
//				std::cout<<"  h2 = "<<h2<<std::endl;
//				std::cout<<"  area2 = "<<area2<<std::endl;
				double overlap = returnAbsoluteOverlappedArea(xMin1,yMin1,w1,h1,xMin2,yMin2,w2,h2);
//				std::cout<<"  overlapped area = "<<overlap<<std::endl;
				double fraction1 = overlap/area1;
				double fraction2 = overlap/area2;
//				std::cout<<"  fractionOfOverlappedArea["<<p<<"] = "<<fraction1<<std::endl;
//				std::cout<<"  fractionOfOverlappedArea["<<h<<"] = "<<fraction2<<std::endl;
				if (fraction1>=minOverlap || fraction2>=minOverlap){
					overlappingWindows[p].push_back(h);
//					std::cout<<"  window "<<h<<" -> overlap = "<<overlap<<std::endl;
				}
			}
		}
	}

//	std::cout<<"Defining a graph object..."<<std::endl;
	Graph g(boxes.size(), overlappingWindows);
	int maxConnComp = boxes.size();
	std::vector<std::vector<int>> connComp(maxConnComp);
//	std::cout<<"Launching the getConnectedComponents() function..."<<std::endl;
	int actualNumOfConnComp = 0;
	g.getConnectedComponents(connComp, actualNumOfConnComp);
//	g.printConnComp(connComp, actualNumOfConnComp);

	std::vector<double> xMin(actualNumOfConnComp);
	std::vector<double> xMax(actualNumOfConnComp);
	std::vector<double> yMin(actualNumOfConnComp);
	std::vector<double> yMax(actualNumOfConnComp);

	for (int p = 0; p < actualNumOfConnComp; p++) {
//		std::cout<<"connComp["<<p<<"].size() = "<<connComp[p].size()<<std::endl;
		for (int h = 0; h < connComp[p].size(); h++) {
//			std::cout<<"    h = "<<h<<std::endl;
			int currBox = connComp[p][h];
//			std::cout<<"    currBox = "<<currBox<<std::endl;
			if (h==0){
				xMin[p] = boxes[currBox][0];
				xMax[p] = boxes[currBox][0] + boxes[currBox][2];
				yMin[p] = boxes[currBox][1];
				yMax[p] = boxes[currBox][1] + boxes[currBox][3];
			}else{
				double currBoxXMin = boxes[currBox][0];
				double currBoxXMax = boxes[currBox][0] + boxes[currBox][2];
				double currBoxYMin = boxes[currBox][1];
				double currBoxYMax = boxes[currBox][1] + boxes[currBox][3];
//				std::cout<<"currBoxXMin = "<<currBoxXMin<<std::endl;
//				std::cout<<"currBoxXMax = "<<currBoxXMax<<std::endl;
//				std::cout<<"currBoxYMin = "<<currBoxYMin<<std::endl;
//				std::cout<<"currBoxYMax = "<<currBoxYMax<<std::endl;

				if (currBoxXMin < xMin[p])
					xMin[p] =  currBoxXMin;
				if (currBoxXMax > xMax[p])
					xMax[p] = currBoxXMax;
				if (currBoxYMin < yMin[p])
					yMin[p] = currBoxYMin;
				if (currBoxYMax > yMax[p])
					yMax[p] = currBoxYMax;
			}
//			std::cout<<"xMin["<<p<<"] = "<<xMin[p]<<std::endl;
//			std::cout<<"xMax["<<p<<"] = "<<xMax[p]<<std::endl;
//			std::cout<<"yMin["<<p<<"] = "<<yMin[p]<<std::endl;
//			std::cout<<"yMax["<<p<<"] = "<<yMax[p]<<std::endl;
		}
	}

//	std::cout<<"Returning the final boxes..."<<std::endl;
	std::vector<std::vector<double>> finalBoxes(actualNumOfConnComp);
	for (int p = 0; p < actualNumOfConnComp; p++) {
		double w = xMax[p] - xMin[p];
		double h = yMax[p] - yMin[p];
		finalBoxes[p].push_back(xMin[p]);
		finalBoxes[p].push_back(yMin[p]);
		finalBoxes[p].push_back(w);
		finalBoxes[p].push_back(h);
	}
	return finalBoxes;
}

void refineBox(int i, int j, int w, int h, double& xMin, double& yMin, double& wNew, double& hNew, int n, int stepSize, std::vector<std::vector<cv::KeyPoint>> allKeypoints){
	//std::cout<<"Refining a bounding box..."<<std::endl;
//	std::cout<<"..._allKeypoints.size() = "<<allKeypoints.size()<<std::endl;
//	std::cout<<"...i = "<<i<<std::endl;
//	std::cout<<"...j = "<<j<<std::endl;
//	std::cout<<"...w = "<<w<<std::endl;
//	std::cout<<"...h = "<<h<<std::endl;
	std::vector<cv::KeyPoint> allKeypointsSingleVector;
	for (int p = n*i; p < n*(i+h); p = p+n) {
		//std::cout<<"  p = "<<p<<std::endl;
		for (int q = j; q < j+w; q++) {
			//std::cout<<"    q = "<<q<<std::endl;
			for (int k = 0; k < allKeypoints[p+q].size(); k++) {
				allKeypointsSingleVector.push_back(allKeypoints[p+q][k]);
				//std::cout<<"allKeypointsSingleVector["<<k<<"].pt.x = "<<allKeypointsSingleVector[k].pt.x<<std::endl;
				//std::cout<<"allKeypointsSingleVector["<<k<<"].pt.y = "<<allKeypointsSingleVector[k].pt.y<<std::endl;
			}
		}
	}
	//std::cout<<"   ...collected all the keypoints of the current bounding box..."<<std::endl;
	//std::cout<<"   ...allKeypointsSingleVector.size() = "<<allKeypointsSingleVector.size()<<std::endl;
	int xMax;
	int yMax;
	for (int k = 0; k < allKeypointsSingleVector.size(); k++) {
		if (k==0){
			xMin = allKeypointsSingleVector[k].pt.x;
			xMax = allKeypointsSingleVector[k].pt.x;
			yMin = allKeypointsSingleVector[k].pt.y;
			yMax = allKeypointsSingleVector[k].pt.y;
		}else{
//			std::cout<<"xMin = "<<xMin<<std::endl;
//			std::cout<<"xMax = "<<xMax<<std::endl;
//			std::cout<<"yMin = "<<yMin<<std::endl;
//			std::cout<<"yMax = "<<yMax<<std::endl;
//			std::cout<<"allKeypointsSingleVector[k].pt.x = "<<allKeypointsSingleVector[k].pt.x<<std::endl;
//			std::cout<<"allKeypointsSingleVector[k].pt.y = "<<allKeypointsSingleVector[k].pt.y<<std::endl;
			if (allKeypointsSingleVector[k].pt.x < xMin)
				xMin = floor(allKeypointsSingleVector[k].pt.x);
			if (allKeypointsSingleVector[k].pt.x > xMax)
				xMax = ceil(allKeypointsSingleVector[k].pt.x);
			if (allKeypointsSingleVector[k].pt.y < yMin)
				yMin = floor(allKeypointsSingleVector[k].pt.y);
			if (allKeypointsSingleVector[k].pt.y > yMax)
				yMax = ceil(allKeypointsSingleVector[k].pt.y);
		}
	}
//	std::cout<<"xMinOld = "<<j*stepSize<<std::endl;
//	std::cout<<"yMinOld = "<<i*stepSize<<std::endl;
//	std::cout<<"wOld = "<<w<<std::endl;
//	std::cout<<"hOld = "<<h<<std::endl;

	wNew = xMax-xMin;
	hNew = yMax-yMin;

//	std::cout<<"xMinNew = "<<xMin<<std::endl;
//	std::cout<<"yMinNew = "<<yMin<<std::endl;
//	std::cout<<"wNew = "<<wNew<<std::endl;
//	std::cout<<"hNew = "<<hNew<<std::endl;
//	std::cout<<"   box refined."<<std::endl;
}

void resizeBox(double& xMin, double& yMin, double& w, double& h, int resizeFactor){
	xMin = resizeFactor*xMin;
	yMin = resizeFactor*yMin;
	w = resizeFactor*w;
	h = resizeFactor*h;
}

void computeBoWsOfBlocks(Codebook c, double ratioThreshold, double distThreshold, std::vector<cv::Mat> _allDescriptors, std::vector<cv::Mat>& BoWsOfAllBlocks){
	// Window number to be modified with block number
	std::cout<<"Preliminary computation of the BoWs of all blocks..."<<std::endl;
	std::cout<<"Block ";
	for (int i = 0; i < _allDescriptors.size(); i++) {
		std::string s = std::to_string(i + 1) + "/" + std::to_string(_allDescriptors.size());
		std::cout << s;
		std::cout << std::string(s.length(), '\b');
		int blockNumber = i;
		cv::Mat BoWOfCurrBlock;
		if (!_allDescriptors[i].empty())
			BoWOfCurrBlock = returnBoWFromDescriptors(_allDescriptors[i], c, blockNumber, ratioThreshold, distThreshold);
		else
			BoWOfCurrBlock = cv::Mat(1, c.returnCodebookSize(), CV_32F, cv::Scalar::all(0));
		BoWOfCurrBlock.copyTo(BoWsOfAllBlocks[i]);
	}
	std::cout<<"\nBoWs computation completed."<<std::endl;
}

cv::Mat returnBoWOFCurrentWindow(std::list<int> windowBlockIndeces, std::vector<cv::Mat> BoWsOfAllBlocks){
	//std::cout<<"Computing the BoW of the current window..."<<std::endl;
	cv::Mat resultingBoW(1, BoWsOfAllBlocks[0].cols, CV_32F, cv::Scalar::all(0));
	for (std::list<int>::iterator it = windowBlockIndeces.begin(); it != windowBlockIndeces.end(); it++){
		resultingBoW = resultingBoW +  BoWsOfAllBlocks[*it];
		//cv::add(resultingBoW, BoWsOfAllBlocks[*it], resultingBoW);
	}
	std::list<int>::iterator it = windowBlockIndeces.begin();
	//showBoW(resultingBoW,*it);
	return resultingBoW;
}

void pruneBoxes(std::vector<std::vector<double>>& boxes, double overlappingRatio){
	int nBoxes = boxes.size();
	std::vector<int> toBeRemoved;
	for (int i = nBoxes -1; i >=0 ; i--) {
		std::vector<double> currBox = boxes[i];
		int si = currBox[4];
		double xi = currBox[0];
		double yi = currBox[1];
		double wi = currBox[2];
		double hi = currBox[3];
		double currArea = (xi+wi)*(yi+hi);
		double overlappedArea = 0;
		//std::cout<<"si = "<<si<<std::endl;
		//std::cout<<"xi = "<<xi<<", "<<"yi = "<<yi<<", "<<"wi = "<<wi<<", "<<"hi = "<<hi<<std::endl;
		//std::cout<<"currArea = "<<currArea<<std::endl;
		for (int j = 0; j < i; j++) {
			std::vector<double> supportBox = boxes[j];
			int sj = supportBox[4];
			double xj = supportBox[0];
			double yj = supportBox[1];
			double wj = supportBox[2];
			double hj = supportBox[3];
			//std::cout<<"    sj = "<<sj<<std::endl;
			//std::cout<<"    xj = "<<xj<<", "<<"yj = "<<yj<<", "<<"wj = "<<wj<<", "<<"hj = "<<hj<<std::endl;
			if (sj==si-1){
				overlappedArea = overlappedArea + returnAbsoluteOverlappedArea(xi,yi,wi,hi,xj,yj,wj,hj);
			}
		}
		//std::cout<<"    overlappedArea = "<<overlappedArea<<std::endl;
		//std::cout<<"    overlappedArea/currArea = "<<overlappedArea/currArea<<std::endl;
		if (si>0 && overlappedArea>0) //  && (overlappedArea/currArea)<overlappingRatio
			toBeRemoved.push_back(i);
	}
	std::vector<std::vector<double>> newBoxes(boxes.size()-toBeRemoved.size());
	int k = 0;
	for (int i = 0; i < boxes.size(); i++) {
		if(std::find(toBeRemoved.begin(), toBeRemoved.end(),i)==toBeRemoved.end()) {
			newBoxes[k] = boxes[i];
			k++;
		}
	}
	boxes = newBoxes;
}

void saveWindow(int key, int& wait, int i, int j, int windowSide, int stepSize, cv::Mat inputImage){
	//std::cout<<"key = "<<key<<std::endl;
	if (key==102)
		wait = 0;
	if (key==118)
		wait = 1;
	if (key==115){
		double x = j*stepSize;
		double y = i*stepSize;
		double w = windowSide;
		double h = windowSide;
		cv::Rect roi(x,y,w,h);
		cv::String path = "C:\\Users\\Francesko\\Desktop\\CVProject\\boatDetector\\cmake-build-release\\SavedImages";
		bool check = checkPath(path);
		if (!check) {
			std::cout<<"Creating folder: "<<path<<std::endl;
			std::system(("mkdir "+path).c_str());
		}
		cv::String name = "C:\\Users\\Francesko\\Desktop\\CVProject\\boatDetector\\cmake-build-release\\SavedImages\\img-w["+std::to_string((i))+","+std::to_string((j))+"].jpg";
		cv::imwrite(name, inputImage(roi));
	}
}

std::vector<std::vector<double>> extractFinalSameLevelNotMergedBoxes(std::vector<std::vector<double>>finalBoxesOfAllLayers, int level){
	std::vector<std::vector<double>> finalSameLevelNotMergedBoxes;
	//std::cout<<"Extracting final not merged boxes of level "<<level<<std::endl;
	for (int i = 0; i < finalBoxesOfAllLayers.size(); i++) {
		if(finalBoxesOfAllLayers[i][4]==level) {
			//std::cout<<"i = "<<i<<std::endl;
			finalSameLevelNotMergedBoxes.push_back(finalBoxesOfAllLayers[i]);
		}
	}
	return finalSameLevelNotMergedBoxes;
}

void readInputImageGTCoords(std::vector<std::vector<double>>& GTCoords, double wRatio, double hRatio){
	bool stop;
	std::string txtFileName;
	std::ifstream txtFile;
	stop = false;
	while(!stop){
		std::cout<<"Please insert the path of the .txt file containing the ground truth coordinates of the boats:"<<std::endl;
		std::cin >> txtFileName;
		std::string str = txtFileName.substr(txtFileName.size()-4,4);
		if (str!=".txt")
			std::cout<<"Error: the file \""<<txtFileName<<"\" is not a .txt file."<<std::endl;
		else{
			txtFile.open(txtFileName);
			if (!txtFile.is_open() || str!=".txt"){
				std::cout<<"Error: the file \""<<txtFileName<<"\" cannot be found."<<std::endl;
			} else
				stop = true;
		}
	}
	char c;
	int nBoats = 0;
	while (1){
		//std::cout<<"----------    Image ["<<imgIdx<<"]    ----------"<<std::endl;
		int spaces = 0;
		int k = 0;
		std::string s  = "";
		c = txtFile.get();
		bool goOn = true;
		std::vector<int> xyCoords;
		while (goOn){
			if (c!=' ')
				s = s+c;
			else{
				if (spaces == 0) {
					s = "";
				}
				if (spaces == 1){
					nBoats = std::stoi(s);
					//std::cout << "nBoats = "<<nBoats<<std::endl;
					s = "";
				}
				if (spaces > 1){
					xyCoords.push_back(std::stoi(s));
					//std::cout << "xyCoords[" << imgIdx << "]["<<k<<"] = " << std::stoi(s)<< std::endl;
					k++;
					s = "";
				}
				spaces = spaces + 1;
			}
			if (c=='\n'){
				xyCoords.push_back(std::stoi(s));
				//std::cout << "xyCoords[" << imgIdx << "]["<<k<<"] = " << std::stoi(s)<< std::endl;
				k++;
				s = "";
				goOn = false;
			}
			c = txtFile.get();
			//std::cout << c;
		}
		std::vector<int> currWindowCorners(xyCoords.size());
		currWindowCorners = xyCoords;
		for (int i = 0; i < 2*2*nBoats-1; i=i+4) {
			//std::cout<<"currWindowsCorners ["<<i<<"] = "<<currWindowCorners[i]<<std::endl;
			//std::cout<<"currWindowsCorners ["<<i+1<<"] = "<<currWindowCorners[i+1]<<std::endl;
			double x = currWindowCorners[i]*wRatio;
			double y = currWindowCorners[i+1]*hRatio;
			double w = currWindowCorners[i+2]*wRatio;
			double h = currWindowCorners[i+3]*hRatio;
			std::vector<double> box = {x,y,w,h};
			GTCoords.push_back(box);
//			std::cout<<"x ["<<i/4<<"] = "<<x<<std::endl;
//			std::cout<<"y ["<<i/4<<"] = "<<y<<std::endl;
//			std::cout<<"w ["<<i/4<<"] = "<<w<<std::endl;
//			std::cout<<"h ["<<i/4<<"] = "<<h<<std::endl;
		}
		if (txtFile.eof()){
			//std::cout<<"EOF"<<std::endl;
			txtFile.close();
			break;
		}
	}
	std::cout<<"GT coordinates of the input image read."<<std::endl;
}

std::vector<std::vector<std::vector<double>>> returnUnionPoligon(std::vector<std::vector<double>> finalBoxes, std::vector<std::vector<double>> GTCoords){
	std::vector<std::vector<double>> allBoxes;
	allBoxes.insert(allBoxes.end(), finalBoxes.begin(), finalBoxes.end());
	allBoxes.insert(allBoxes.end(), GTCoords.begin(), GTCoords.end());
	std::vector<std::vector<int>> idxBoxesWithOverlap(allBoxes.size());

	for (int i = 0; i < allBoxes.size(); i++) {
		double xMin1 = allBoxes[i][0];
		double yMin1 = allBoxes[i][1];
		double w1 = allBoxes[i][2];
		double h1 = allBoxes[i][3];
		for (int j = 0; j < allBoxes.size(); j++) {
			if (j!=i){
				double xMin2 = allBoxes[j][0];
				double yMin2 = allBoxes[j][1];
				double w2 = allBoxes[j][2];
				double h2 = allBoxes[j][3];
				double overlappedArea = returnAbsoluteOverlappedArea(xMin1, yMin1, w1, h1, xMin2, yMin2, w2, h2);
				if (overlappedArea>0) {
					idxBoxesWithOverlap[i].push_back(j);
				}
			}
		}
	}

	//std::cout<<"Defining a graph object..."<<std::endl;
	Graph g(allBoxes.size(), idxBoxesWithOverlap);
	int maxConnComp = idxBoxesWithOverlap.size();
	std::vector<std::vector<int>> connComp(maxConnComp);
	//std::cout<<"Launching the getConnectedComponents() function..."<<std::endl;
	int actualNumOfConnComp = 0;
	g.getConnectedComponents(connComp, actualNumOfConnComp);
	//g.printConnComp(connComp, actualNumOfConnComp);

	std::vector<std::vector<std::vector<double>>> poligonsNodes(actualNumOfConnComp);
	for (int i = 0; i < actualNumOfConnComp; i++) {
		for (int j = 0; j < connComp[i].size(); j++) {
			double xMin = allBoxes[connComp[i][j]][0];
			double yMin = allBoxes[connComp[i][j]][1];
			double w = allBoxes[connComp[i][j]][2];
			double h = allBoxes[connComp[i][j]][3];
			double xMax = xMin + w;
			double yMax = yMin + h;

			std::vector<std::vector<double>> points(4);

			points[0] = {xMin,yMin};
			points[1] = {xMin,yMax};
			points[2] = {xMax,yMax};
			points[3] = {xMax,yMin};

			for (int k = 0; k < 4; k++) {
				//std::cout<<"Point [x,y] = "<<"["<<points[k][0]<<", "<<points[k][1]<<"] lies inside a box:" << liesInsideABox(points[k], allBoxes)<< std::endl;
				if (!liesInsideABox(points[k], allBoxes))
					poligonsNodes[i].push_back(points[k]);
			}
		}
	}

	//std::cout<<"Creating separated vectors with overlapping boxes..."<<std::endl;
	for (int i = 0; i < actualNumOfConnComp; i++) {
		//std::cout<<"Connected component number "<<i<<std::endl;
		std::vector<std::vector<double>> overlappingBoxes(connComp[i].size());
		for (int j = 0; j < connComp[i].size(); j++) {
			overlappingBoxes[j] = allBoxes[connComp[i][j]];
//			double xMin = overlappingBoxes[j][0];
//			double yMin = overlappingBoxes[j][1];
//			double w = overlappingBoxes[j][2];
//			double h = overlappingBoxes[j][3];
//			std::cout<<"  xMin = "<<xMin<<std::endl;
//			std::cout<<"  yMin = "<<yMin<<std::endl;
//			std::cout<<"  w = "<<w<<std::endl;
//			std::cout<<"  h = "<<h<<std::endl;
		}
		std::vector<std::vector<double>> intersections = returnIntersectionPoints(overlappingBoxes);
		for (int k = 0; k < intersections.size(); k++) {
			//std::cout<<"Intersection [x,y] = "<<"["<<intersections[k][0]<<", "<<intersections[k][1]<<"] lies inside a box:" << liesInsideABox(intersections[k], allBoxes)<< std::endl;
			if (!liesInsideABox(intersections[k], allBoxes))
				poligonsNodes[i].push_back(intersections[k]);
		}
	}
	removeDuplicates(poligonsNodes);
//	for (int i = 0; i < poligonsNodes.size(); i++) {
//		std::cout<<"Number of final nodes of polygon "<<i<<" = "<<poligonsNodes[i].size()<<std::endl;
//		for (int m = 0; m < poligonsNodes[i].size(); m++) {
//			std::cout<<"  [x, y]["<<m<<"] = ["<<poligonsNodes[i][m][0]<<", "<<poligonsNodes[i][m][1]<<"]"<<std::endl;
//		}
//	}
	sortNodes(poligonsNodes);
	return poligonsNodes;
}

void removeDuplicates(std::vector<std::vector<double>>& points){
	//std::cout<<"Removing duplicates..."<<std::endl;
	std::vector<std::vector<double>> newPoints;
	for (int i = 0; i < points.size(); i++) {
		double currX = points[i][0];
		double currY = points[i][1];
		std::vector<double> currPoint = {currX, currY};
		if (newPoints.empty()){
			newPoints.push_back(currPoint);
		} else {
			bool present = false;
			for (int j = 0; j < newPoints.size() && !present; j++) {
				if (currX==newPoints[j][0] && currY==newPoints[j][1])
					present = true;
			}
			if (!present)
				newPoints.push_back(currPoint);
		}
	}
//	std::cout<<"NEW POINTS WITHOUT DUPLICATES"<<std::endl;
//	for (int i = 0; i < newPoints.size(); i++) {
//		std::cout<<"  [x, y]["<<i<<"] = ["<<newPoints[i][0]<<", "<<newPoints[i][1]<<"]"<<std::endl;
//	}
	points = newPoints;
}

void sortNodes(std::vector<std::vector<std::vector<double>>>& poligonsNodes){
	//std::cout<<"Start sorting all nodes for drawing... "<<std::endl;
	int numOfConnComp = poligonsNodes.size();
	std::vector<std::vector<std::vector<double>>> tmpNodes(numOfConnComp);
	//std::cout<<"Num of final poligons to draw = "<<numOfConnComp<<std::endl;
	int lastMovement = 0; //1 down, 2 right, 3 up, 4,left
	for (int i = 0; i < numOfConnComp; i++) {
		//std::cout<<"--------------------      Polygon "<<i<<"      --------------------"<<std::endl;
		std::vector<std::vector<double>> currCompNodes = poligonsNodes[i];
		std::vector<int> visited(currCompNodes.size(), 0);
		//std::cout<<"Num of nodes = "<<currCompNodes.size()<<std::endl;
		std::vector<double> currNode(2);
		double currX;
		double currY;
		int currIdx;
		double x0;
		double y0;
		int idx0;
		for (int j = 0; j <= currCompNodes.size(); j++) {
			//std::cout<<"Iteration j = "<<j<<":"<<std::endl;
			bool addedANode = false;
			if (j==0){
				currIdx = getTopLeftNodeIdx(poligonsNodes[i]);
				//std::cout<<"  TopLeftNodeIndex = "<<currIdx<<std::endl;
				currX = poligonsNodes[i][currIdx][0];
				currY = poligonsNodes[i][currIdx][1];
				currNode = {currX,currY};
				x0 = currX;
				y0 = currY;
				idx0 = currIdx;
				visited[currIdx]=1;
				tmpNodes[i].push_back(currNode);
				//std::cout<<"  x0 = "<<currX<<std::endl;
				//std::cout<<"  y0 = "<<currY<<std::endl;
			} else {
				// Searching same x, greater y
				//std::cout<<"  currX = "<<currX<<std::endl;
				//std::cout<<"  currY = "<<currY<<std::endl;
				//std::cout<<"  currIdx = "<<currIdx<<std::endl;

				// Searching down
				std::vector<std::vector<double>> candidates1;
				std::vector<std::vector<double>> candidates2;
				std::vector<std::vector<double>> candidates3;
				std::vector<std::vector<double>> candidates4;
				int m = 0;
				for (int k = 0; k < currCompNodes.size() && !addedANode; k++) {
					double xK = poligonsNodes[i][k][0];
					double yK = poligonsNodes[i][k][1];
					if (k!=currIdx && xK==currX && yK>currY && (!visited[k] || (visited[k] && k==idx0))) {
						//std::cout<<"  [x, y] = [" << xK << ", " << yK << "] --> down candidate"<<std::endl;
						std::vector<double> point {xK,yK};
						candidates1.push_back(point);
					}
					if (k!=currIdx && yK==currY && xK>currX && (!visited[k] || (visited[k] && k==idx0))) {
						//std::cout<<"  [x, y] = [" << xK << ", " << yK << "] --> right candidate"<<std::endl;
						std::vector<double> point {xK,yK};
						candidates2.push_back(point);
					}
					if (k!=currIdx && xK==currX && yK < currY && (!visited[k] || (visited[k] && k==idx0))) {
						//std::cout<<"  [x, y] = [" << xK << ", " << yK << "] --> up candidate"<<std::endl;
						std::vector<double> point {xK,yK};
						candidates3.push_back(point);
					}
					if (k!=currIdx && yK==currY && xK<currX && (!visited[k] || (visited[k] && k==idx0))) {
						//std::cout<<"  [x, y] = [" << xK << ", " << yK << "] --> left candidate"<<std::endl;
						std::vector<double> point {xK,yK};
						candidates4.push_back(point);
					}
				}

//				std::cout<<"  Down candidates = "<<candidates1.size()<<std::endl;
//				std::cout<<"  Right candidates = "<<candidates2.size()<<std::endl;
//				std::cout<<"  Up candidates = "<<candidates3.size()<<std::endl;
//				std::cout<<"  Left candidates = "<<candidates4.size()<<std::endl;

				if (lastMovement==1){
					if (!candidates4.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates4,lastMovement,4,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					} else if (!candidates1.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates1,lastMovement,1,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates2.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates2,lastMovement,2,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					} else if (!candidates3.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates3,lastMovement,3,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}
				} else if (lastMovement==2 || lastMovement==0){
					if (!candidates1.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates1,lastMovement,1,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					} else if (!candidates2.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates2,lastMovement,2,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates3.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates3,lastMovement,3,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates4.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates4,lastMovement,4,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}
				}  else if (lastMovement==3){
					if (!candidates2.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates2,lastMovement,2,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					} else if (!candidates3.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates3,lastMovement,3,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates4.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates4,lastMovement,4,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates1.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates1,lastMovement,1,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}
				}  else if (lastMovement==4){
					if (!candidates3.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates3,lastMovement,3,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					} else if (!candidates4.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates4,lastMovement,4,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates1.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates1,lastMovement,1,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}else if (!candidates2.empty()){
						addedANode = true;
						moveAhead(currNode,i,visited,tmpNodes,currIdx,poligonsNodes,candidates2,lastMovement,2,addedANode);
						currX = currNode[0];
						currY = currNode[1];
					}
				}
			}
		}
//		for (int n = 0; n < visited.size(); n++) {
//			std::cout << "visited["<<n<<"] = "<< visited[n] << std::endl;
//		}
	}

	poligonsNodes = tmpNodes;
}

bool liesInsideABox(std::vector<double> point, std::vector<std::vector<double>> boxes){
	bool response = false;
	double xPoint = point[0];
	double yPoint = point[1];

	for (int i = 0; i < boxes.size() && !response; i++) {
		double xMin = boxes[i][0];
		double yMin = boxes[i][1];
		double w = boxes[i][2];
		double h = boxes[i][3];
		double xMax = xMin + w;
		double yMax= yMin + h;
		if (xMin < xPoint && xPoint < xMax && yMin < yPoint && yPoint < yMax){
			response = true;
		}
	}
	return response;
}

std::vector<std::vector<double>> returnIntersectionPoints(std::vector<std::vector<double>> overlappingBoxes){
	//std::cout<<"Computing the intersection points..."<<std::endl;
	std::vector<std::vector<double>> intersections;
	for (int i = 0; i < overlappingBoxes.size(); i++) {
		//std::cout<<"i = "<<i<<std::endl;
		double xMin1 = overlappingBoxes[i][0];
		double yMin1 = overlappingBoxes[i][1];
		double w1 = overlappingBoxes[i][2];
		double h1 = overlappingBoxes[i][3];
		double xMax1 = xMin1 + w1;
		double yMax1 = yMin1 + h1;
//		std::cout<<"[xMin1, yMin1]["<<i<<"] = ["<<xMin1<<", "<<yMin1<<"]"<<std::endl;
//		std::cout<<"[xMin1, yMax1]["<<i<<"] = ["<<xMin1<<", "<<yMax1<<"]"<<std::endl;
//		std::cout<<"[xMax1, yMax1]["<<i<<"] = ["<<xMax1<<", "<<yMax1<<"]"<<std::endl;
//		std::cout<<"[xMax1, yMin1]["<<i<<"] = ["<<xMax1<<", "<<yMin1<<"]"<<std::endl;
		for (int j = 0; j < overlappingBoxes.size(); j++) {
			//std::cout<<"  j = "<<j<<std::endl;
			double xMin2 = overlappingBoxes[j][0];
			double yMin2 = overlappingBoxes[j][1];
			double w2 = overlappingBoxes[j][2];
			double h2 = overlappingBoxes[j][3];
			double xMax2 = xMin2 + w2;
			double yMax2 = yMin2 + h2;
			if (j!=i){
//				std::cout<<"   [xMin2, yMin2]["<<j<<"] = ["<<xMin2<<", "<<yMin2<<"]"<<std::endl;
//				std::cout<<"   [xMin2, yMax2]["<<j<<"] = ["<<xMin2<<", "<<yMax2<<"]"<<std::endl;
//				std::cout<<"   [xMax2, yMax2]["<<j<<"] = ["<<xMax2<<", "<<yMax2<<"]"<<std::endl;
//				std::cout<<"   [xMax2, yMin2]["<<j<<"] = ["<<xMax2<<", "<<yMin2<<"]"<<std::endl;
				double xSideStart = xMin1 <= xMin2? xMin2 : xMin1;
				double xSideEnd = xMax1 <= xMax2? xMax1 : xMax2;
				double ySideStart = yMin1 <= yMin2? yMin2 : yMin1;
				double ySideEnd = yMax1 <= yMax2? yMax1 : yMax2;
				double xSide = xSideEnd - xSideStart;
				double ySide = ySideEnd - ySideStart;
				if (xSide>0 && ySide>0){
					std::vector<double> p1 = {xSideStart,ySideStart};
					std::vector<double> p2 = {xSideStart,ySideEnd};
					std::vector<double> p3 = {xSideEnd,ySideEnd};
					std::vector<double> p4 = {xSideEnd,ySideStart};
//					std::cout<<"  => intersection box:"<<std::endl;
//					std::cout<<"     p1 = [xMin, yMin] = ["<<xSideStart<<", "<<ySideStart<<"]"<<std::endl;
//					std::cout<<"     p2 = [xMin, yMax] = ["<<xSideStart<<", "<<ySideEnd<<"]"<<std::endl;
//					std::cout<<"     p3 = [xMax, yMax] = ["<<xSideEnd<<", "<<ySideEnd<<"]"<<std::endl;
//					std::cout<<"     p4 = [xMax, yMin] = ["<<xSideEnd<<", "<<ySideStart<<"]"<<std::endl;
					intersections.push_back(p1);
					intersections.push_back(p2);
					intersections.push_back(p3);
					intersections.push_back(p4);
				}
			}
		}
	}
	removeDuplicates(intersections);

//	std::cout<<"Final intersection points found:"<<std::endl;
//	for (int i = 0; i < intersections.size(); i++) {
//		std::cout<<"  intersections["<<i<<"] = ["<<intersections[i][0]<<", "<<intersections[i][1]<<"]"<<std::endl;
//	}

	//std::cout<<"Intersection points computed."<<std::endl;
	return intersections;
}

void removeDuplicates(std::vector<std::vector<std::vector<double>>>& allPoints){
	for (int k = 0; k < allPoints.size(); k++) {
		std::vector<std::vector<double>> points = allPoints[k];
		std::vector<std::vector<double>> newPoints;
		for (int i = 0; i < points.size(); i++) {
			double currX = points[i][0];
			double currY = points[i][1];
			std::vector<double> currPoint = {currX, currY};
			if (newPoints.empty()){
				newPoints.push_back(currPoint);
			} else {
				bool present = false;
				for (int j = 0; j < newPoints.size() && !present; j++) {
					if (currX==newPoints[j][0] && currY==newPoints[j][1])
						present = true;
				}
				if (!present)
					newPoints.push_back(currPoint);
			}
		}
		allPoints[k] = newPoints;
	}
//	std::cout<<"NEW POINTS WITHOUT DUPLICATES"<<std::endl;
//	for (int i = 0; i < newPoints.size(); i++) {
//		std::cout<<"  [x, y]["<<i<<"] = ["<<newPoints[i][0]<<", "<<newPoints[i][1]<<"]"<<std::endl;
//	}

}

std::vector<double> getMinDistPoint(std::vector<double> currPoint, std::vector<std::vector<double>> candidates){
	double dist;
	double idx;
	for (int k = 0; k < candidates.size(); k++) {
		double xDiff = currPoint[0]-candidates[k][0];
		double yDiff = currPoint[1]-candidates[k][1];
		if (k==0) {
			dist = std::sqrt(std::pow(xDiff, 2) + std::pow(yDiff, 2));
			idx = 0;
		}else{
			double currDist = std::sqrt(std::pow(xDiff, 2) + std::pow(yDiff, 2));
			if (currDist < dist){
				dist = currDist;
				idx = k;
			}
		}
	}
	std::vector<double> result(3);
	result[0] = candidates[idx][0];
	result[1] = candidates[idx][1];
	return result;
}

int returnIndex(std::vector<double> point, std::vector<std::vector<double>> pointsVec){
//	std::cout<<"Returning currIdx..."<<std::endl;
//	std::cout<<"point[0] = "<<point[0]<<std::endl;
//	std::cout<<"point[1] = "<<point[1]<<std::endl;
	for (int i = 0; i < pointsVec.size(); i++) {
//		std::cout<<"pointsVec["<<i<<"][0] = "<<pointsVec[i][0]<<std::endl;
//		std::cout<<"pointsVec["<<i<<"][1] = "<<pointsVec[i][1]<<std::endl;
		if (pointsVec[i][0]==point[0] && pointsVec[i][1]==point[1]){
			//std::cout<<"  currIdx found = "<<i<<std::endl;
			return i;
		}
	}
	return -1;
}

int getTopLeftNodeIdx(std::vector<std::vector<double>> pointsVec){
	int idx;
	double x;
	double y;
	for (int i = 0; i < pointsVec.size(); i++) {
		double currX = pointsVec[i][0];
		double currY = pointsVec[i][1];
		if (i==0){
			x = currX;
			y = currY;
			idx = 0;
		}else{
			if((currX<x && currY<y) || (currX<x && currY==y) || (currX==x && currY<y)){
				x = currX;
				y = currY;
				idx = i;
			}
		}
	}
	return idx;
}

void moveAhead(std::vector<double>& currNode, int i, std::vector<int>& visited, std::vector<std::vector<std::vector<double>>>& tmpNodes,
			   int& currIdx, std::vector<std::vector<std::vector<double>>> poligonsNodes, std::vector<std::vector<double>> candidates, int& lastMovement,
			   int currMovement, bool& addedANode){
	currNode = getMinDistPoint(currNode, candidates);
	std::string s = "";
	if (currMovement==1){
		s = "down";
	} else if (currMovement==2){
		s = "right";
	} else if (currMovement==3){
		s = "up";
	} else if (currMovement==4){
		s = "left";
	}
	//std::cout << "  ==> moving "<< s << ": new currNode --> [currX, currY] = [" << currNode[0] << ", " << currNode[1] << "]" << std::endl;
	currIdx = returnIndex(currNode, poligonsNodes[i]);
	//std::cout << "  currIdx = " << currIdx << std::endl;
	//std::cout << "  visited["<<currIdx<<"] = "<<1<< std::endl;
	visited[currIdx] = 1;
	tmpNodes[i].push_back(currNode);
	addedANode = true;
	lastMovement = currMovement;
}

double returnArea(std::vector<std::vector<std::vector<double>>> polygon){
	std::vector<double> areas(polygon.size(),0);
	for (int i = 0; i < polygon.size(); i++) {
		std::vector<std::vector<double>> currVertices = polygon[i];
		int j = currVertices.size()-1;
		for (int k = 0; k < currVertices.size(); k++) {
			double xJ = currVertices[j][0];
			double yJ = currVertices[j][1];
			double xK = currVertices[k][0];
			double yK = currVertices[k][1];
			areas[i] = areas[i] + (xJ+xK)*(yJ-yK);
			j = k;
		}
		areas[i] = abs(areas[i]/2.0);
		//std::cout<<"Area polygon "<<i<<" = "<<areas[i]<<std::endl;
	}
	double finalArea = 0;
	for (int i = 0; i < areas.size(); i++) {
		finalArea = finalArea + areas[i];
	}
	//std::cout<<"Final area = "<<finalArea<<std::endl;
	return finalArea;
}

std::vector<std::vector<std::vector<double>>> returnIntersectionPoligon(std::vector<std::vector<double>> finalBoxes, std::vector<std::vector<double>> GTCoords){
	std::vector<std::vector<double>> allBoxes;
	allBoxes.insert(allBoxes.end(), finalBoxes.begin(), finalBoxes.end());
	allBoxes.insert(allBoxes.end(), GTCoords.begin(), GTCoords.end());
	std::vector<std::vector<int>> idxBoxesWithOverlap(allBoxes.size());
	std::vector<std::vector<double>> intermediateIntersections;
	std::vector<std::vector<std::vector<double>>> intersections;
	for (int i = 0; i < finalBoxes.size(); i++) {
		double xMin1 = finalBoxes[i][0];
		double yMin1 = finalBoxes[i][1];
		double w1 = finalBoxes[i][2];
		double h1 = finalBoxes[i][3];
		double xMax1 = xMin1 + w1;
		double yMax1 = yMin1 + h1;
//		std::cout<<"[xMin1, yMin1]["<<i<<"] = ["<<xMin1<<", "<<yMin1<<"]"<<std::endl;
//		std::cout<<"[xMin1, yMax1]["<<i<<"] = ["<<xMin1<<", "<<yMax1<<"]"<<std::endl;
//		std::cout<<"[xMax1, yMax1]["<<i<<"] = ["<<xMax1<<", "<<yMax1<<"]"<<std::endl;
//		std::cout<<"[xMax1, yMin1]["<<i<<"] = ["<<xMax1<<", "<<yMin1<<"]"<<std::endl;
		for (int j = 0; j < GTCoords.size(); j++) {
			double xMin2 = GTCoords[j][0];
			double yMin2 = GTCoords[j][1];
			double w2 = GTCoords[j][2];
			double h2 = GTCoords[j][3];
			double xMax2 = xMin2 + w2;
			double yMax2 = yMin2 + h2;
//			std::cout<<"   [xMin2, yMin2]["<<j<<"] = ["<<xMin2<<", "<<yMin2<<"]"<<std::endl;
//			std::cout<<"   [xMin2, yMax2]["<<j<<"] = ["<<xMin2<<", "<<yMax2<<"]"<<std::endl;
//			std::cout<<"   [xMax2, yMax2]["<<j<<"] = ["<<xMax2<<", "<<yMax2<<"]"<<std::endl;
//			std::cout<<"   [xMax2, yMin2]["<<j<<"] = ["<<xMax2<<", "<<yMin2<<"]"<<std::endl;
			double xSideStart = xMin1 <= xMin2? xMin2 : xMin1;
			double xSideEnd = xMax1 <= xMax2? xMax1 : xMax2;
			double ySideStart = yMin1 <= yMin2? yMin2 : yMin1;
			double ySideEnd = yMax1 <= yMax2? yMax1 : yMax2;
			double xSide = xSideEnd - xSideStart;
			double ySide = ySideEnd - ySideStart;
			if (xSide>0 && ySide>0) {
//				std::vector<double> p1 = {xSideStart,ySideStart};
//				std::vector<double> p2 = {xSideStart,ySideEnd};
//				std::vector<double> p3 = {xSideEnd,ySideEnd};
//				std::vector<double> p4 = {xSideEnd,ySideStart};
//				std::vector<std::vector<double>> boxVertices = {p1,p2,p3,p4,p1};
//				std::cout<<"  => intersection box:"<<std::endl;
//				std::cout<<"     p1 = [xMin, yMin] = ["<<xSideStart<<", "<<ySideStart<<"]"<<std::endl;
//				std::cout<<"     p2 = [xMin, yMax] = ["<<xSideStart<<", "<<ySideEnd<<"]"<<std::endl;
//				std::cout<<"     p3 = [xMax, yMax] = ["<<xSideEnd<<", "<<ySideEnd<<"]"<<std::endl;
//				std::cout<<"     p4 = [xMax, yMin] = ["<<xSideEnd<<", "<<ySideStart<<"]"<<std::endl;
//				intersections.push_back(boxVertices);
				std::vector<double> boxXYWH = {xSideStart, ySideStart, xSide, ySide};
				intermediateIntersections.push_back(boxXYWH);
			}
		}
	}
	intersections = returnUnionPoligon(intermediateIntersections, intermediateIntersections);
	return intersections;
}

// ----------   DEBUG SECTION   ----------
void printImageFeatureBlocks(std::vector<std::vector<cv::KeyPoint>> allKeypoints, std::vector<cv::Mat> allDescriptors, int n, int stepSize){
	std::cout<<"Starting the printing of the feature blocks..."<<std::endl;
	int currXMin = 0;
	int currXMax = 0;
	int currYMin = 0;
	int currYMax = 0;
	int i = 0;
	int j = 0;
	for (int k = 0; k < allKeypoints.size(); k++) {
		i = (k - (k%n))/n;
		j = k%n;
		currXMin = j*stepSize;
		currXMax = (j+1)*stepSize;
		currYMin = i*stepSize;
		currYMax = (i+1)*stepSize;
		std::cout<<"Keypoints in the block [xMin, xMax]x[yMin, yMax] = ["<<currXMin<<","<<currXMax<<"]x["<<currYMin<<","<<currYMax<<"]:"<<std::endl;
		for (int l = 0; l < allKeypoints[n*i+j].size(); l++) {
			std::cout<<"    keypoint["<<l<<"] = ["<<allKeypoints[n*i+j][l].pt.x<<", "<<allKeypoints[n*i+j][l].pt.y<<"];"<<std::endl;
			if (currXMin > allKeypoints[n*i+j][l].pt.x || allKeypoints[n*i+j][l].pt.x > currXMax || currYMin > allKeypoints[n*i+j][l].pt.y || allKeypoints[n*i+j][l].pt.y > currYMax){
				std::cout<<"Error: current keypoint not in the proper block."<<std::endl;
				//exit(1);
			}
		}
	}
}

void printWindowFeatureBlocks(std::vector<std::vector<cv::KeyPoint>> currKeypoints, std::vector<cv::Mat> currDescriptors, int i, int j, int l, int stepSize){
	std::cout<<"Starting the printing of the window feature blocks..."<<std::endl;
	int currXMin = j*stepSize;
	int currXMax = (j+1)*stepSize;
	int currYMin = i*stepSize;
	int currYMax = (i+1)*stepSize;
	int k = 0;
	for (int jj = 0; jj < l; jj++) {
		for (int ii = 0; ii < l; ii++) {
			std::cout<<"Keypoints in the block [xMin, xMax]x[yMin, yMax] = ["<<currXMin<<","<<currXMax<<"]x["<<currYMin<<","<<currYMax<<"]:"<<std::endl;
			for (int p = 0; p < currKeypoints[k].size(); p++) {
				std::cout<<"    keypoint["<<p<<"] = ["<<currKeypoints[k][p].pt.x<<", "<<currKeypoints[k][p].pt.y<<"];"<<std::endl;
				if (currXMin > currKeypoints[k][p].pt.x || currKeypoints[k][p].pt.x > currXMax || currYMin > currKeypoints[k][p].pt.y || currKeypoints[k][p].pt.y > currYMax){
					std::cout<<"Error: current keypoint not in the proper block."<<std::endl;
					//exit(1);
				}
			}
			k++;
			currYMin += stepSize;
			currYMax += stepSize;
		}
		currYMin = i*stepSize;
		currYMax = (i+1)*stepSize;
		currXMin += stepSize;
		currXMax += stepSize;
	}
}

void printListElements(std::list<int> list){
	std::cout<<"Blocks of the initial window (nodes of the list): "<<std::endl;
	int i=0;
	for (std::list<int>::iterator it = list.begin(); it != list.end(); it++) {
		int elm = *it;
		std::cout<<"   node["<<i<<"] = image block "<<elm<<";"<<std::endl;
		i++;
	}
}

void printDescriptors(std::string s, cv::Mat descriptors){
	std::ofstream txtFile;
	txtFile.open(s);
	if (!txtFile.is_open()){
		std::cout<<"Error: the descriptor.txt cannot be found. "<<std::endl;
		exit(1);
	}
	for (int i = 0; i < descriptors.rows; i++) {
		txtFile << "D"<<i<<" = [";
		for (int j = 0; j < descriptors.cols; j++) {
			txtFile << descriptors.at<float>(i,j) << ", ";
		}
		txtFile << "...]\n";
	}
}

void removeZeroRows(cv::Mat inputMat, cv::Mat& outputMat){
	for (int i = 0; i < inputMat.rows; i++) {
		cv::Rect row(0, i, inputMat.cols, 1);
		if (!inputMat(row).empty()){
			cv::Mat nonZeroRow;
			inputMat(row).copyTo(nonZeroRow);
			outputMat.push_back(nonZeroRow);
		}
	}
}

void showBoW(cv::Mat inputBoW, int BoWnum){
	cv::Mat BoWToDisplay(30, inputBoW.cols, CV_8U, cv::Scalar::all(255));
	for (int i = 0; i < inputBoW.cols; i++) {
		float h = inputBoW.at<float>(0,i);
		if (h>0)
			cv::line(BoWToDisplay, cv::Point(i,0), cv::Point(i,h),cv::Scalar(0,0,0),2);
	}
	cv::flip(BoWToDisplay, BoWToDisplay, 0);
	cv::resize(BoWToDisplay, BoWToDisplay, cv::Size(1000,400));
	cv::imwrite("testBoWs\\BoW"+std::to_string(BoWnum)+".jpg", BoWToDisplay);
}