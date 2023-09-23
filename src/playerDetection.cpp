#include "playerDetection.h"

using cv::Mat;

playerDetection::playerDetection(Mat input_image){
    input_image_ = input_image;

}

void playerDetection::startprocess(){

    cv::Mat input_clone = input_image_.clone();

    Mat blurred;
    cv::bilateralFilter(input_clone, blurred, 5, 1000, 400);
    cv::GaussianBlur(blurred,blurred, cv::Size(9,9), 2);

    cv::Mat colorSupp = colorSuppression(blurred, 10);
    output_img = input_image_.clone();

    cv::HOGDescriptor hog;

    hog.load("../model/model.yml");

    Mat img_gray;
    cvtColor(blurred, img_gray, cv::COLOR_BGR2GRAY);

    // Apply local contrast enhancement to the grayscale image using CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(img_gray, img_gray);

    int exp = 6;
    int winSizeX = pow(2, exp);
    int winSizeY = pow(2, exp + 1);

    std::vector<cv::Rect> all_rects;

    // Ensure that winSizeX and winSizeY are multiples of blockStride
    if (winSizeX <= img_gray.cols && winSizeY <= img_gray.rows) {
        hog.winSize = cv::Size(winSizeX, winSizeY);
        std::vector<cv::Rect> rects;
        std::vector<double> weights;


        for (float scale = 1; scale >= 0.3; scale -= 0.1){

            cv::Mat resized_img;
            cv::resize(img_gray, resized_img, cv::Size(), scale, scale);

            hog.detectMultiScale(resized_img, rects, weights, 0, cv::Size(2, 2), cv::Size(15,15), 1.01, 1, false);

            for(cv::Rect element : rects){
                // Adjust the rectangles to the original image size
                element.x /= scale;
                element.y /= scale;
                element.width /= scale;
                element.height /= scale;

                all_rects.push_back(element);
            }
        }
    }

    groupRectangles(all_rects, 2, 0.6);
    for (size_t i = 0; i < all_rects.size(); i++)
    {
        std::cout <<  all_rects[i].x << " " << all_rects[i].y << " " << all_rects[i].width << " " << all_rects[i].height << std::endl;
        cv::rectangle(output_img, all_rects[i], cv::Scalar(0, 255, 0), 2);
    }
}


cv::Mat playerDetection::getOutput() const {
    return output_img;
}