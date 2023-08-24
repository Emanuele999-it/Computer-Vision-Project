#include "imageProcessing.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


// remove
#include "MeanShift.h"

using cv::Mat;
using std::vector;

cv::RNG rng(1000);

imageProcessing::imageProcessing(Mat input_image){
    input_image_=input_image;
}


void imageProcessing::startprocess(){
    Mat field = fieldSegmentation();

}


Mat imageProcessing::fieldSegmentation(){

    displayMat(input_image_, "input image");

    // FIRST TRY

    // Reduce noise to input image + maintain edges
    Mat filtered_image_;
    cv::bilateralFilter(input_image_,filtered_image_,21,3,3);

    // Find edges
    Mat cannyOut;
    cv::Canny(filtered_image_,cannyOut,100,100);

    // Find contour
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    findContours(cannyOut, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(cannyOut.size(), CV_8UC3);

    // Different colors for the edges
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
    }

    displayMat(drawing, "drawing");
    return cannyOut;


/*
    // SECOND TRY


    cvtColor(input_image_, input_image_, COLOR_RGB2Lab);

    MeanShift MSProc(8, 16);
    // MSProc.MSFiltering(image);
    // Segmentation Process include Filtering Process (Region Growing)
    MSProc.MSSegmentation(input_image_);

    cout << "\nthe Spatial Bandwith is " << MSProc.hs;
    cout << "\nthe Color Bandwith is " << MSProc.hr;

    cvtColor(input_image_, input_image_, COLOR_Lab2RGB);

    displayMat(input_image_);

    return input_image_;
*/


/*
    // THIRD TRY

    // Reduce noise to input image + maintain edges
    Mat filtered_image_;
    int edgeSize = 3;
    //cv::GaussianBlur(input_image_,filtered_image_, cv::Size(edgeSize,edgeSize),10,10);
    cv::bilateralFilter(input_image_,filtered_image_,edgeSize,10,10);

    // Reshape the image to a 2D matrix of pixels (rows × columns, 3 color channels)
    Mat reshapedImage = filtered_image_.reshape(1, filtered_image_.rows * filtered_image_.cols);
    
    // Convert the image to float for k-means clustering
    reshapedImage.convertTo(reshapedImage, CV_32F);

    // Number of clusters for k-means
    int numClusters = 2;

    // Criteria for k-means algorithm
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0);

    // K-means clustering
    Mat labels, centers;
    kmeans(reshapedImage, numClusters, labels, criteria, numClusters, KMEANS_PP_CENTERS, centers);

    // Convert the cluster centers to 8-bit unsigned integers
    centers.convertTo(centers, CV_8U);

    // Map labels to colors and create segmented image
    Mat segmented = Mat::zeros(input_image_.size(), input_image_.type());
    for (int i = 0; i < input_image_.rows; ++i) {
        for (int j = 0; j < input_image_.cols; ++j) {
            int label = labels.at<int>(i * input_image_.cols + j);
            segmented.at<Vec3b>(i, j) = centers.at<Vec3b>(label);
        }
    }

    displayMat(segmented);

    return segmented;
*/
}