#ifndef CUSTOM_HOUGH_DETECTOR_H
#define CUSTOM_HOUGH_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "displayMat.h"

class customHoughDetector{

    private:
        cv::Mat input_image_;
        cv::Mat result_img_;

    public:
        customHoughDetector(cv::Mat input_image);
        void startprocess();
        cv::Mat colorSoppression(cv::Mat img);
};

#endif // CUSTOM_HOUGH_DETECTOR_H