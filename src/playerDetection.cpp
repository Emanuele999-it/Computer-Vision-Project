// EMANUELE PASE 2097904

#include "playerDetection.h"

using cv::Mat;

playerDetection::playerDetection(const Mat & input_image){
    input_image_ = input_image;

}

void playerDetection::startprocess(){

    cv::Mat input_clone = input_image_.clone();

    Mat blurred;
    cv::bilateralFilter(input_clone, blurred, 5, 1000, 400);
    cv::GaussianBlur(blurred,blurred, cv::Size(9,9), 2);

    cv::Mat colorSupp = colorSuppression(blurred, 5);
    output_img = input_image_.clone();

    cv::HOGDescriptor hog;

    // Load model
    hog.load("../model/model.yml");

    Mat img_gray;
    cvtColor(blurred, img_gray, cv::COLOR_BGR2GRAY);

    int exp = 6;
    int winSizeX = pow(2, exp);
    int winSizeY = pow(2, exp + 1);

    std::vector<cv::Rect> all_rects;

    hog.winSize = cv::Size(winSizeX, winSizeY);
    std::vector<cv::Rect> rects;
    std::vector<double> weights;

    // Scaling + detectionMultiscaling
    for (float scale = 1; scale >= 0.8; scale -= 0.06)
    {

        cv::Mat resized_img;
        cv::resize(img_gray, resized_img, cv::Size(), scale, scale);

        hog.detectMultiScale(resized_img, rects, weights, 0, cv::Size(4, 4), cv::Size(30, 30), 1.05, 3, false);

        for (cv::Rect element : rects)
        {
            // Adjust the rectangles to the original image size
            element.x /= scale;
            element.y /= scale;
            element.width /= scale;
            element.height /= scale;

            all_rects.push_back(element);
        }
    }

    // Remove similar rectangles
    groupRectangles(all_rects, 2, 0.3);

    std::vector<int> teams = getPlayerTeam(input_clone, &all_rects);

    for (size_t i = 0; i < all_rects.size(); i++)
    {
        std::cout <<  all_rects[i].x << " " << all_rects[i].y << " " << all_rects[i].width << " " << all_rects[i].height << " " << teams[i] <<std::endl;
        cv::rectangle(output_img, all_rects[i], cv::Scalar(0, 255, 0), 2);
    }
}


cv::Mat playerDetection::getOutput() const {
    return output_img;
}