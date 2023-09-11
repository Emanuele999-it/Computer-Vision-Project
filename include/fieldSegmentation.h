#ifndef FIELD_SEGMENTATION_H
#define FIELD_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "displayMat.h"
#include "colorSuppression.h"
#include "mostFrequentColorFiltering.h"

class fieldSegmentation{

    private:
        cv::Mat input_image_;
        cv::Mat result_image_;
        //std::vector<std::vector<cv::Point>> contours_vec_;

        cv::Mat maskGeneration(cv::Mat img);
        void noiseReduction(cv:: Mat img);

    public:
        fieldSegmentation(cv::Mat input_image);
        void startprocess();

        cv::Mat returnInputImage() const;
        cv::Mat returnResultImage() const;
};

#endif // FIELD_SEGMENTATION_H