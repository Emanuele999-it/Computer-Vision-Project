#include "displayMat.h"

// Allow to visualize images with an inline function
const void displayMat(const cv::Mat &toDisplay, std::string name, int waitValue){
    cv::namedWindow(name, cv::WINDOW_NORMAL);
	cv::imshow(name, toDisplay);
	cv::waitKey(waitValue);
}