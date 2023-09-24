// EMANUELE PASE 2097904

#include "playerTeam.h"

const std::vector<int> getPlayerTeam(const cv::Mat &img, const std::vector<cv::Rect> *rectangles){
   
    std::vector<cv::Mat> subImages;
    std::vector<int> teams;

    // Step 1: Extract sub-images outlined by rectangles
    for (const cv::Rect& rect : *rectangles) {
        // Extract the sub-image using a region of interest (ROI)
        cv::Mat subImage = img(rect);

        // Define a rectangle around the human figure to initialize GrabCut
        cv::Rect rectangle(0, 0, subImage.cols - 1 , subImage.rows - 1);

        // Initialize mask and grabCut parameters
        cv::Mat mask(subImage.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
        mask(rectangle).setTo(cv::Scalar(cv::GC_PR_FGD));  // Set rectangle region as probable foreground

        cv::Mat bgdModel, fgdModel;

        // Apply GrabCut algorithm with the mean-shifted image
        cv::grabCut(subImage, mask, rectangle, bgdModel, fgdModel, 75, cv::GC_EVAL);

        // Modify the mask to consider probable and definite foreground as foreground
        cv::Mat resultMask = (mask == cv::GC_PR_FGD /* | mask == cv::GC_PR_FGD*/);

        // Create a binary mask for visualization
        cv::Mat binaryMask;
        resultMask.convertTo(binaryMask, CV_8U);

        // Create a masked output image
        cv::Mat outputImage;
        subImage.copyTo(outputImage, binaryMask);

        subImages.push_back(outputImage);
    }


    // Step 2: Convert each image to grayscale
    std::vector<cv::Mat> grayImages;
    for (const cv::Mat& img : subImages) {
        cv::Mat gray;
        cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        grayImages.push_back(gray);
    }

    // Step 3: Calculate the average intensity of all images
    double totalIntensity = 0.0;
    for (const cv::Mat& grayImg : grayImages) {
        cv::Scalar avgIntensity = mean(grayImg);
        totalIntensity += avgIntensity[0];
    }

    double averageIntensity = totalIntensity / grayImages.size();

    // Step 4: Iterate images and compare them to average intensity
    for (const cv::Mat& grayImg : grayImages) {
        cv::Scalar avgIntensitySingle = mean(grayImg);
        // Compare the average intensity of the single image with the average intensity of all images
        if (avgIntensitySingle[0] < averageIntensity) {
            teams.push_back(1);
        } else {
            teams.push_back(2);
        }
    }

    return teams;
}