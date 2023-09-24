// AHMED AKYOL 2049673

#ifndef PLAYER_SEGMENTATION_H
#define PLAYER_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

#include "displayMat.h"
#include "colorSuppression.h"
#include "mostFrequentColorFiltering.h"

class playerSegmentation{

    private:
        cv::Mat input_image_;
        cv::Mat field_segmentation_;
        double calculateaver_intensity(const cv::Mat& image, const cv::Mat& mask);

    public:
        playerSegmentation(const cv::Mat & input_image);
        cv::Mat startprocess();

};

#endif // PLAYER_SEGMENTATION_H