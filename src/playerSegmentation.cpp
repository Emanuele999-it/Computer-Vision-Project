#include "playerSegmentation.h"

using cv::Mat;

playerSegmentation::playerSegmentation(const cv::Mat & input_image, const cv::Mat & fieldSegmentation){
    input_image_ = input_image;
    field_segmentation_ = fieldSegmentation;
}

Mat playerSegmentation::startprocess(){
// --------------------- Remove field form img ----------------------
    // Adapt field-segmentation to player detection

    Mat field = field_segmentation_.clone();
/* 
    // Create a kernel for dilation
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // Create an empty output image for the result
    cv::Mat result1;

    // Apply dilation to each channel separately
    std::vector<cv::Mat> channels;
    cv::split(temp, channels); // Split the image into B, G, and R channels

    for (int i = 0; i < 3; ++i) {
        //cv::morphologyEx(channels[i], channels[i], cv::MORPH_CLOSE, kernel);
        //cv::erode(channels[i], channels[i], cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 1);
        cv::erode(channels[i], channels[i], kernel, cv::Point(-1,-1), 4);
    }
   
    // Merge the channels back into a 3-channel image
    cv::merge(channels, result1); 
*/
    cv::Mat input_clone = input_image_.clone();

    // Convert the mask image to grayscale
    cv::Mat maskGray;
    cv::cvtColor(field, maskGray, cv::COLOR_BGR2GRAY);
    
    // Apply a binary threshold to create a binary mask
    cv::Mat binaryMask1;
    cv::threshold(maskGray, binaryMask1, 1, 255, cv::THRESH_BINARY);

    // Iterate through the pixels of the target image
    for (int y = 0; y < input_clone.rows; y++) {
        for (int x = 0; x < input_clone.cols; x++) {
            if (binaryMask1.at<uchar>(y, x) == 255) {
                // If the pixel in the mask is white (green), set the corresponding pixel in the target image to black
                input_clone.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    displayMat(input_clone, "input clone");


    // ---------------------- Filter by contours ----------------------
/*
    Mat blurred;
    cv::GaussianBlur(input_clone, blurred, cv::Size(3,3),3,3);
    
    cv::Mat gray;
    cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

    //displayMat(gray, "gray");
    
    cv::Mat binary;
    cv::threshold(gray, binary, 20, 200, cv::THRESH_BINARY);
    //displayMat(binary, "binary pre");
    //cv::dilate(binary, binary, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 7);

    //displayMat(binary, "binary post");

    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat result = cv::Mat::zeros(blurred.size(), CV_8UC3); // Create a black image

    double minContourArea = 500; // Set your minimum contour area here

    for (size_t i = 0; i < contours.size(); i++) {
        double contourArea = cv::contourArea(contours[i]);

        // Check if the contour area is greater than the minimum area
        if (contourArea > minContourArea) {
            // Draw the contour on the result image
            cv::drawContours(result, contours, i, cv::Scalar(255, 255, 255), -1); // Fill the contour with white
        }
    }

    //displayMat(result, "filtered by contours");
*/
    // ---------------------- Removed background ----------------------
/*
    // Create a mask for the black regions
    cv::Mat mask2;
    cv::threshold(result, mask2, 1, 255, cv::THRESH_BINARY);

    // Create an output image
    cv::Mat segmentationNoField;

    // Copy the regions from the original image to the output image
    input_image_.copyTo(segmentationNoField, mask2);

    displayMat(segmentationNoField , "test segmentation no filed");

    Mat partial_result_col_suppression = colorSuppression(segmentationNoField,5);

    displayMat(partial_result_col_suppression, "partial res col supp");
*/

    // ---------------------- Grabcut ----------------------------------

    // Apply color quantization
    cv::Mat quantizedImage;
    cv::cvtColor(input_clone, quantizedImage, cv::COLOR_BGR2Lab); // Convert to Lab color space
    quantizedImage.convertTo(quantizedImage, CV_32F); // Convert to float for k-means
    int K = 3; // Number of clusters (adjust as needed)
    cv::Mat reshapedImage = quantizedImage.reshape(1, quantizedImage.rows * quantizedImage.cols);

    // Perform k-means clustering
    cv::Mat labels, centers;
    cv::kmeans(reshapedImage, K, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2), 3, cv::KMEANS_PP_CENTERS, centers);

    // Assign each pixel to the cluster center color
    cv::Mat clusteredLabImage(quantizedImage.size(), quantizedImage.type());
    for (int i = 0; i < quantizedImage.rows * quantizedImage.cols; ++i) {
        int clusterIdx = labels.at<int>(i);
        cv::Vec3f& color = centers.at<cv::Vec3f>(clusterIdx);
        clusteredLabImage.at<cv::Vec3f>(i) = color;
    }

    // Convert the clustered Lab image back to BGR
    cv::Mat clusteredImage;
    clusteredLabImage.convertTo(clusteredImage, CV_8U);
    cv::cvtColor(clusteredImage, clusteredImage, cv::COLOR_Lab2BGR);

    // Define a rectangle around the human figure to initialize GrabCut
    cv::Rect rectangle(50,50, input_image_.cols-50, input_image_.rows-50);

    // Initialize mask and grabCut parameters
    cv::Mat mask(input_image_.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    mask(rectangle).setTo(cv::Scalar(cv::GC_PR_FGD));  // Set rectangle region as probable foreground

    cv::Mat bgdModel, fgdModel;

    // Apply GrabCut algorithm with the clustered image
    cv::grabCut(clusteredImage, mask, rectangle, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT);

    // Modify the mask to consider probable and definite foreground as foreground
    cv::Mat resultMask = (mask == cv::GC_PR_FGD) | (mask == cv::GC_FGD);

    // Create a binary mask for visualization
    cv::Mat binaryMask;
    resultMask.convertTo(binaryMask, CV_8U);

    // Create a masked output image
    cv::Mat outputImage;
    input_image_.copyTo(outputImage, binaryMask);

    // Display the original and segmented images side by side
    cv::Mat combinedImage;
    cv::hconcat(input_image_, outputImage, combinedImage);


    return combinedImage;
}