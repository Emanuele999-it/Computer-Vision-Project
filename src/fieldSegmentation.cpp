#include "fieldSegmentation.h"

using cv::Mat;

fieldSegmentation::fieldSegmentation(const cv::Mat & input_image){
    input_image_ = input_image;
}

void fieldSegmentation::startprocess()
{
    // Reduce noise to input image
    cv::Mat resultBilateral, resultGaussian;
    cv::bilateralFilter(input_image_,resultBilateral,5,3000,40);
    cv::GaussianBlur(resultBilateral,resultGaussian, cv::Size(5,5),3,3);

    //displayMat(resultGaussian, "gaussian res", 1);

    // clusterize by similar colors
    cv::Mat colorSuppressed = colorSuppression(resultGaussian, 4);

    //displayMat(input_image_, "input image", 1);
    //displayMat(colorSuppressed, "quantized_image",1);

    Mat mfc = mostFrequentColorFiltering(colorSuppressed, resultGaussian);

    //displayMat(mfc, "mostFreqColor",1);

    noiseReduction(mfc);

    //displayMat(result_image_, "risultato");
}


Mat fieldSegmentation::maskGeneration(const Mat & img){

  // Filtering operation
  const int kNeighborhoodDiamter = 5;
  const int kSigmaColor = 1000;
  const int kSigmaSpace = 200;


  //bilateralFilter(img, input_image_, kNeighborhoodDiamter, kSigmaColor, kSigmaSpace);
  input_image_ = img.clone();
  // Asphalt and sky mask_ generation
  Mat mask;
  const cv::Scalar kLowTreshColor = cv::Scalar(100, 100, 100);
  const cv::Scalar kHighTreshColor = cv::Scalar(255, 255, 255);
  inRange(input_image_, kLowTreshColor, kHighTreshColor, mask);

  // Apply mask_ to the input image
  const cv::Scalar kAssignedScalar = cv::Scalar(255, 255, 255);
  input_image_.setTo(kAssignedScalar, mask);
  
  return mask;
}



void fieldSegmentation::noiseReduction(const cv::Mat & img){
    // Create a kernel for dilation
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // Create an empty output image for the result
    cv::Mat result1;

    // Apply dilation to each channel separately
    std::vector<cv::Mat> channels;
    cv::split(img, channels); // Split the image into B, G, and R channels

    for (int i = 0; i < 3; ++i) {
        //cv::morphologyEx(channels[i], channels[i], cv::MORPH_CLOSE, kernel);
        //cv::erode(channels[i], channels[i], cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 1);
        cv::erode(channels[i], channels[i], kernel, cv::Point(-1,-1), 3);
        cv::dilate(channels[i], channels[i], kernel, cv::Point(-1,-1), 4);

    }

    // Merge the channels back into a 3-channel image
    cv::merge(channels, result1); 

    //displayMat(result1, "after morph modif", 1);

    // Convert the image to HSV color space
    Mat hsv;
    cvtColor(result1, hsv, cv::COLOR_BGR2HSV);

    // Define the range of green color in HSV
    cv::Scalar lower_green(35, 50, 50); // Adjust these values as needed
    cv::Scalar upper_green(85, 255, 255); // Adjust these values as needed

    // Create a mask to identify green regions
    Mat green_mask;
    inRange(hsv, lower_green, upper_green, green_mask);

    // Find connected components (blobs) in the green regions
    std::vector<std::vector<cv::Point>> contours;
    findContours(green_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find the main green region by selecting the largest blob
    double maxArea = 0;
    int mainGreenIdx = -1;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            mainGreenIdx = static_cast<int>(i);
        }
    }

    // Create a new mask to combine the main green region and nearby blobs
    Mat result_mask = Mat::zeros(img.size(), CV_8U);

    for (size_t i = 0; i < contours.size(); i++) {
        if (i == mainGreenIdx) {
            // Include the main green region
            drawContours(result_mask, contours, i, cv::Scalar(255), cv::FILLED);
        } 
        else {
            // Calculate the distance between the current blob and the main green region
            double distance = matchShapes(contours[mainGreenIdx], contours[i], cv::CONTOURS_MATCH_I2, 0);

            // If the distance is below a threshold, include the blob
            if (distance < 4.75) {
                drawContours(result_mask, contours, i, cv::Scalar(255), cv::FILLED);
            }
        }
    }

    // Apply the mask to the original image to suppress unwanted green regions
    Mat result;
    img.copyTo(result, result_mask);

    result_image_ = result;
}


cv::Mat fieldSegmentation::returnInputImage() const {
    cv::Mat temp = input_image_.clone();
    return temp;
}

cv::Mat fieldSegmentation::returnResultImage() const {
    cv::Mat temp = result_image_.clone();
    return temp;
}