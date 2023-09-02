#ifndef FIELD_SEGMENTATION_H
#define FIELD_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "displayMat.h"

class fieldSegmentation{

    private:
        cv::Mat input_image_;
        cv::Mat result_img_;
        cv::Mat markers_image_;
        cv::Mat mask_;
        std::vector<std::vector<cv::Point>> contours_vec_;

        void preProcess();
        void postProcess();
        cv::Mat colorSuppression(cv::Mat img, int k); 
        cv::Mat maskGeneration(cv::Mat img);
        cv::Mat mostFrequentColorFiltering(const cv::Mat img);

    public:
        fieldSegmentation(cv::Mat input_image);
        void startprocess();

        cv::Mat returnInputImage();
        cv::Mat returnResultImage();
        cv::Mat returnMaskImage();
};

#endif // FIELD_SEGMENTATION_H