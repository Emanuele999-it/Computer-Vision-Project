// EMANUELE PASE 2097904

#ifndef IMGPROCESSING_H
#define IMGPROCESSING_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>

#include "displayMat.h"
#include "fieldSegmentation.h"
#include "playerSegmentation.h"
#include "playerDetection.h"


class imageProcessing{

    private:
        cv::Mat input_image_;

    public:
        imageProcessing(const cv::Mat & input_image);
        void startprocess();
};

#endif