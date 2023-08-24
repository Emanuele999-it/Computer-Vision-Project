#ifndef DISPLAYMAT__H
#define DISPLAYMAT__H

#include <opencv2/opencv.hpp>

// Allow to visualize images with an inline function
void displayMat(const cv::Mat &toDisplay, std::string name = "noName", int waitValue = 0);
	

#endif // DISPLAYMAT__H