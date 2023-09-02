#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include "displayMat.h"

class imageProcessing{

    private:
        cv::Mat input_image_;

    public:
        imageProcessing(cv::Mat input_image);
        void startprocess();
};

#endif