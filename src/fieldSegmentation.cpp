#include "fieldSegmentation.h"

using cv::Mat;

fieldSegmentation::fieldSegmentation(cv::Mat input_image){
    input_image_ = input_image;
}

void fieldSegmentation::startprocess()
{
    // Reduce noise to input image
    cv::Mat resultBilateral, resultGaussian;
    cv::bilateralFilter(input_image_,resultBilateral,11,15,15);
    cv::GaussianBlur(resultBilateral,resultGaussian, cv::Size(5,5),3,3);

    // clusterize by similar colors
    cv::Mat colorSuppressed = colorSuppression(resultGaussian, 3);

    displayMat(input_image_, "input image", 1);
    displayMat(colorSuppressed, "quantized_image",1);

    Mat mfc = mostFrequentColorFiltering(colorSuppressed);

    displayMat(mfc, "mostFreqColor",1);

    noiseReduction(mfc);

    displayMat(result_image_, "risultato");
}

// -----------------------------------------------------------------------------
/*

        Alternatives for solving the filed segmentation problems:
            1 - compare the result with the original image to understand if the region is the same
            2 - if a black shape is surrounded by green shapes (all 4 sides) -> connect
            3 - merge result with different color suppressed images
            4 - dilate / erode to connect dark areas

*/
// -----------------------------------------------------------------------------



// allow to reduce the number of colors
cv::Mat fieldSegmentation::colorSuppression(cv::Mat img, int k){ // k = number of color quantization
    
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

Mat fieldSegmentation::maskGeneration(Mat img){

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



cv::Mat fieldSegmentation::mostFrequentColorFiltering(const cv::Mat img){
    // Reshape the image to a list of pixels
    Mat reshaped = img.reshape(1, img.rows * img.cols);

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
    Mat output = Mat::zeros(img.size(), img.type());
    double tolerance = 80;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            if (cv::norm(img.at<cv::Vec3b>(i, j), mostFrequentColor) <= tolerance ) {
                output.at<cv::Vec3b>(i, j) = desiredColor;
            }
        }
    }

    return output;
}


void fieldSegmentation::noiseReduction(cv::Mat img){
    // Create a kernel for dilation
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    // Create an empty output image for the result
    cv::Mat result1;

    // Apply dilation to each channel separately
    std::vector<cv::Mat> channels;
    cv::split(img, channels); // Split the image into B, G, and R channels

    for (int i = 0; i < 3; ++i) {
        //cv::morphologyEx(channels[i], channels[i], cv::MORPH_CLOSE, kernel);
        cv::erode(channels[i], channels[i], cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 3);

    }

    // Merge the channels back into a 3-channel image
    cv::merge(channels, result1);
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
        } else {
            // Calculate the distance between the current blob and the main green region
            double distance = matchShapes(contours[mainGreenIdx], contours[i], cv::CONTOURS_MATCH_I2, 0);

            // If the distance is below a threshold, include the blob
            if (distance < 3.95) {
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