#include <highgui.hpp>
#include <imgproc.hpp>
#include <imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <opencv2/core/utils/filesystem.hpp>

class Codebook {
public:
	Codebook(int codebookSize, int descriptorSize, std::vector<cv::Mat>& positiveExamplesDescriptors);
	Codebook(int codebookSize, int descriptorSize, std::string existingCodebook);
	int returnCodebookSize();
	int returnDescriptorSize();
	cv::Mat returnCodebook();
private:
	int mCodebookSize;
	int mDescriptorSize;
	cv::Mat mCodebook;
	cv::String selectCodebook();
	void loadImages(std::vector<cv::Mat>& images, std::string s);
	bool checkPath(const std::string &s);
	void extractBoatExamples(std::vector<cv::Mat> codebookImages, std::vector<cv::Mat>& boatExamples, std::vector<std::vector<int>>& gtCoordinates);
	void readGroundTruthCoordsFromTxT(std::vector<cv::Mat> codebookImages, std::vector<std::vector<int>>& gtCoordinates);
	cv::Mat computeFeatureDescriptors(std::vector<cv::Mat>& examples, std::vector<cv::Mat>& descriptors, std::vector<std::vector<int>>& gtCoordinates);
	void performKMeans(cv::Mat allDescriptors, cv::Mat& centroids);
	void writeMatToTxt(cv::Mat M, std::string s);
	cv::Mat readCodebookFromTxt(std::string s);
	cv::Mat convert2cvMat(std::vector<std::vector<float>> v);
	void printPositiveExamplesForDebug(std::vector<cv::Mat> inputImages, std::vector<std::vector<int>> gtCoordinates);
};
