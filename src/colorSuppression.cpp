#include "colorSuppression.h"

// Allow to reduce the number of colors
const cv::Mat colorSuppression(const cv::Mat & img, int k){ // k = number of color quantization
    
    cv::Mat reshaped_image = img.reshape(1, img.rows * img.cols); 

    cv::Mat labels, centers;
    cv::Mat reshaped_image_float;
    reshaped_image.convertTo(reshaped_image_float, CV_32F); 

    cv::kmeans(reshaped_image_float, k, labels, cv::TermCriteria(), 10, cv::KMEANS_RANDOM_CENTERS, centers); 

    cv::Mat quantized_image(img.size(), img.type());
    for (int i = 0; i < reshaped_image.rows; ++i) {
        quantized_image.at<cv::Vec3b>(i) = centers.at<cv::Vec3f>(labels.at<int>(i));
    }

    return quantized_image;
}

