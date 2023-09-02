#include "imageProcessing.h"
#include "fieldSegmentation.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


using cv::Mat;
using std::vector;

cv::RNG rng(1000);

imageProcessing::imageProcessing(Mat input_image){
    input_image_=input_image;
}


void imageProcessing::startprocess(){
    fieldSegmentation field(input_image_);
    field.startprocess();
}


