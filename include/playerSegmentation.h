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

    public:
        playerSegmentation(const cv::Mat & input_image, const cv::Mat & fieldSegmentation);
        cv::Mat startprocess();

};

#endif // PLAYER_SEGMENTATION_H