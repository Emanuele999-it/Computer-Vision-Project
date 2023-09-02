#include "fieldSegmentation.h"

using cv::Mat;

fieldSegmentation::fieldSegmentation(cv::Mat input_image){
    input_image_ = input_image;
}

void fieldSegmentation::startprocess()
{
    // Reduce noise to input image + maintain edges
    cv::Mat resultBilateral, resultGaussian;
    cv::bilateralFilter(input_image_,resultBilateral,31,21,21);
    cv::GaussianBlur(resultBilateral,resultGaussian, cv::Size(3,3),3,3);


    cv::Mat colorSuppressed = colorSuppression(resultGaussian, 3);


    displayMat(input_image_, "input image", 1);
    displayMat(colorSuppressed, "quantized_image",1);

    Mat mfc = mostFrequentColorFiltering(colorSuppressed);

    displayMat(mfc, "mostFreqColor");

    mask_ = maskGeneration(mfc); 


    // Perform operations needed for watershed
	preProcess();

    watershed(resultBilateral, markers_image_);

    postProcess();


    displayMat(result_img_, "result");
}


void fieldSegmentation::preProcess(){
    // Distance transform
    Mat dist_image;
    cv::distanceTransform(mask_, dist_image, cv::DIST_L2, 3);


    // Image normalization
    Mat normalized_image;
    const double kLowNorm = 0;
    const double kHighNorm = 1.0;
    normalize(dist_image, normalized_image, kLowNorm, kHighNorm, cv::NORM_MINMAX);


    // Threshold to obtain the markers for waterhseed methods
    Mat thresholded_image;
    const double kLowTresholdNorm = 0.1;
    const double kHighThresholdNorm = 1.0;
	threshold(normalized_image, thresholded_image, kLowTresholdNorm, kHighThresholdNorm, cv::THRESH_BINARY);
    

    // FindContours image needs the CV_8U version of the distance image
	Mat dist_image_8u;
	thresholded_image.convertTo(dist_image_8u, CV_8U);


	// Find the markers
    contours_vec_;
	findContours(dist_image_8u, contours_vec_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


	// Create the markers image for the watershed method
	markers_image_ = Mat::zeros(thresholded_image.size(), CV_32S);
	for (size_t i = 0; i < contours_vec_.size(); i++)
	{
		drawContours(markers_image_, contours_vec_, static_cast<int>(i), cv::Scalar(static_cast<int>(i)+1), -1);
	}
}



void fieldSegmentation::postProcess(){
    //Post-processing 
    Mat markers;
	markers_image_.convertTo(markers, CV_8U);
	bitwise_not(markers, markers);

    // Generate colors
	std::vector<cv::Vec3b> color_tab;
	for (size_t i = 0; i < contours_vec_.size(); i++)
	{
		int b = cv::theRNG().uniform(0, 255);
		int g = cv::theRNG().uniform(0, 255);
		int r = cv::theRNG().uniform(0, 255);
		color_tab.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	// Create the output image
	result_img_ = Mat::zeros(markers_image_.size(), CV_8UC3);

	// Fill labeled objects with random colors
	// Fill labeled objects with random colors
	for (int i = 0; i < markers_image_.rows; i++)
	{
		for (int j = 0; j < markers_image_.cols; j++)
		{
			int index = markers_image_.at<int>(i,j);
			if (index > 0 && index <= static_cast<int>(contours_vec_.size()))
			{
				result_img_.at<cv::Vec3b>(i,j) = color_tab[index-1];
			}
		}
	}
}


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
        colorFreq[pixel]++;
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

    // Create an output image with only the most frequent color
    Mat output = Mat::zeros(img.size(), img.type());
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            if (img.at<cv::Vec3b>(i, j) == mostFrequentColor) {
                output.at<cv::Vec3b>(i, j) = mostFrequentColor;
            }
        }
    }

    return output;
}