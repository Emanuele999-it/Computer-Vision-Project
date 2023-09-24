#include "playerSegmentation.h"

using cv::Mat;

playerSegmentation::playerSegmentation(const cv::Mat & input_image){
    input_image_ = input_image;
}

double playerSegmentation::calculateaver_intensity(const cv::Mat& image, const cv::Mat& mask) {
    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

    cv::Scalar aver_intensity = cv::mean(grayscale, mask);
    return aver_intensity[0];
}


cv::Mat playerSegmentation::startprocess(){
    

    // function to calculate the average pixel intensity

    cv::Mat input = input_image_.clone();

    // mean shift as a color suppression, so that the it is easier to process
    cv::pyrMeanShiftFiltering(input, input, 3, 25, 4); // Spatial Window Size, Color Window Size, Maximum Pyramid Level

    // bilateral filer in order to blur the even areas while keeping the edges
    cv::Mat bilateral_out;
    cv::bilateralFilter(input, bilateral_out, 9, 100, 100); // diameter, sigma color, sigma space
    
    //here initializing grabcut algorithm in order ro detect the players as foreground objects
  
    cv::Rect roi(0, 0, input.cols - 1 , input.rows - 1);

    cv::Mat mask(input.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    mask(roi).setTo(cv::Scalar(cv::GC_PR_FGD)); 

    cv::Mat background, foreground;

    cv::grabCut(bilateral_out, mask, roi, background, foreground, 75, cv::GC_EVAL);

    cv::Mat foreground_mask = (mask == cv::GC_PR_FGD);

    cv::Mat binary_mask;
    foreground_mask.convertTo(binary_mask, CV_8U);

    cv::Mat grabcut_out;
    input.copyTo(grabcut_out, binary_mask);
    
    // canny edge detection for being able to work on edges
    cv::Mat canny_out;
    cv::cvtColor(grabcut_out, canny_out, cv::COLOR_BGR2GRAY);
    cv::Canny(canny_out, canny_out, 50, 100);

    // the straight lines in the images are generally noises from the playground, advertisement panels etc.
    //in order to eliminate these straight lines hough line transform is used.
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(canny_out, lines, 1, CV_PI / 180, 175); 

    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];

        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        double x1 = x0 + 1000 * (-b); 
        double y1 = y0 + 1000 * (a);
        double x2 = x0 - 1000 * (-b); 
        double y2 = y0 - 1000 * (a);

        // black lines are drawn on top of the detected lines, so that we can eliminate them
        cv::line(canny_out, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0), 2);
    }

    // here a bunch of morphological operations are applied. The reason for this to get rid of the thin lines, holes in the 
    // image and other kinds of noises.
    int size = 35;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * size + 1, 2 * size + 1 ));
    cv::dilate(canny_out, canny_out, kernel);
    cv::erode(canny_out, canny_out, kernel);
    
    // vertical closing
    cv::Mat vertical = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 50));
    cv::erode(canny_out, canny_out, vertical);
    cv::dilate(canny_out, canny_out, vertical);
   
    // horizontal closing
    cv::Mat horizontal = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(50, 1));
    cv::erode(canny_out, canny_out, horizontal);
    cv::dilate(canny_out, canny_out, horizontal);


    // contour detection is applied in oreder to fill the black regions inside players caused by morphological opertaions
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_out, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat final_out = cv::Mat::zeros(canny_out.size(), CV_8U);

    for (const auto& i : contours) {
            
      double contourArea = cv::contourArea(i);
        if (contourArea > 1000) {
                cv::drawContours(final_out, std::vector<std::vector<cv::Point>>{i}, 0, cv::Scalar(255), -1);
        }
    }

    // Create a masked output image
    cv::Mat colored_out;
    input.copyTo(colored_out, final_out);

    //TEAM CLASSIFICATION
    cv::Mat result = colored_out.clone();

    std::vector<std::vector<cv::Point>> contour;
    cv::findContours(final_out, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<double> aver_intensities;

    // average grayscale intensity calculation for each region
    for (size_t i = 0; i < contour.size(); i++) {
        cv::Mat playerMask = cv::Mat::zeros(final_out.size(), CV_8UC1);
        cv::drawContours(playerMask, contour, i, cv::Scalar(255), cv::FILLED);
        double averageIntensity = calculateaver_intensity(colored_out, playerMask);
        aver_intensities.push_back(averageIntensity);
    }

    // dynamic threshold calculation
    double dynamic_thr = 0.0;
    for (double intensity : aver_intensities) {
        dynamic_thr += intensity;
    }
    dynamic_thr /= aver_intensities.size();


    for (size_t i = 0; i < contour.size(); i++) {
        cv::Mat player_mask = cv::Mat::zeros(foreground_mask.size(), CV_8UC1);
        cv::drawContours(player_mask, contour, i, cv::Scalar(255), cv::FILLED);
        double aver_intensity = calculateaver_intensity(colored_out, player_mask);

        if (aver_intensity > dynamic_thr) {

            cv::drawContours(result, contour, i, cv::Scalar(0, 0, 255), -1); // first team's color
        }
        else {
            cv::drawContours(result, contour, i, cv::Scalar(255, 0, 0), -1); // second teams color
        }
    }

    field_segmentation_ = result;
    
    return field_segmentation_;

}