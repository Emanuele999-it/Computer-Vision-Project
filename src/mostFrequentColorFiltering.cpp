#include "mostFrequentColorFiltering.h"

// Custom hash function for Vec3b
struct Vec3bHash {
    size_t operator()(const cv::Vec3b& v) const {
        return std::hash<int>()(v[0]) ^ std::hash<int>()(v[1]) ^ std::hash<int>()(v[2]);
    }
};

// Custom equality function for Vec3b
struct Vec3bEqual {
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};


cv::Mat mostFrequentColorFiltering(const cv::Mat & img, const cv::Mat & blurred){
    // Reshape the image to a list of pixels
    cv::Mat reshaped = img.reshape(1, img.rows * img.cols);

    // Count the frequency of each color using an unordered_map with custom hash and equality functions
    std::unordered_map<cv::Vec3b, int, Vec3bHash, Vec3bEqual> colorFreq;
    for (int i = 0; i < reshaped.rows; ++i) {
        cv::Vec3b pixel = reshaped.at<cv::Vec3b>(i, 0);

        // Calculate the sum of color channels (R+G+B)
        int colorSum = pixel[0] + pixel[1] + pixel[2];

        // Skip colors that are too dark based on the thresholdSum
        if (colorSum >= 60) {
            colorFreq[pixel]++;
        }
    }

    // Find the most frequent color
    cv::Vec3b mostFrequentColor;
    int maxFrequency = 0;
    for (const auto& entry : colorFreq) {
        if (entry.second > maxFrequency) {
            maxFrequency = entry.second;
            mostFrequentColor = entry.first;
        }
    }

    cv::Vec3b desiredColor(0, 255, 0);

    // Create an output image with only the regions with the most frequent color
    cv::Mat output = cv::Mat::zeros(img.size(), img.type());
    double tolerance = 70;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {

            cv::Vec3b pixel_img = img.at<cv::Vec3b>(i, j);
            cv::Vec3b pixel_blurred = blurred.at<cv::Vec3b>(i, j);

            if(cv::norm(img.at<cv::Vec3b>(i, j), mostFrequentColor) <= tolerance) {
                output.at<cv::Vec3b>(i,j) = desiredColor;
            }
            /*
            else{
                // giocare con variazione colori per ottenere risultati buoni per immagine 2 e 8 
                double pixel_difference = cv::norm(pixel_blurred, pixel_img);
                if (pixel_difference <= tolerance){
                        output.at<cv::Vec3b>(i,j) = desiredColor;
                }
            }
            */
        }
    }
    return output;
}