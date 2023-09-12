#include "playerDetection.h"

using cv::Mat;

playerDetection::playerDetection(Mat input_image){
    input_image_ = input_image;

}

void ResizeBoxes(cv::Rect& box) {
	box.x += cvRound(box.width * 0.1);
	box.width = cvRound(box.width * 0.8);
	box.y += cvRound(box.height * 0.06);
	box.height = cvRound(box.height * 0.8);
}

void playerDetection::startprocess(){
/*
    cv::CascadeClassifier human_cascade;
    human_cascade.load("../HaarFeatures/fullbody.xml");

    output_img = cv::Mat::zeros(input_image_.size(), CV_8UC3); // Create a black image
    // Load the Haar Cascade XML file for face detection
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("../HaarFeatures/fullbody.xml")) {
        std::cerr << "Error loading Haar Cascade XML file!" << std::endl;
        return input_image_;
    }
    output_img = input_image_.clone();
    cv::Mat gray;
    cv::cvtColor(input_image_, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> humans;
    human_cascade.detectMultiScale(gray, humans, 1.1, 0, 0, cv::Size(40, 25));

    for (const cv::Rect& rect : humans) {
        cv::rectangle(output_img, rect, cv::Scalar(0, 255, 0), 2);
    }

    return output_img;
*/
/*
    output_img = input_image_.clone();

    // Initialize HOG descriptor and use human detection classifier coefficients
	cv::HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	// Detect people and save them to detections
	std::vector<cv::Rect> detections;
	hog.detectMultiScale(input_image_, detections, 0, cv::Size(2, 2), cv::Size(32, 32), 1.2, 2);

	// Resize detection boxes and draw them
	for (auto& detection : detections) {
		ResizeBoxes(detection);
		cv::rectangle(output_img, detection.tl(), detection.br(), cv::Scalar(255, 0, 0), 2);
	}

*/

    cv::Mat input_clone = input_image_.clone();

    Mat blurred;
    cv::bilateralFilter(input_clone, blurred, 3, 40, 11);

    cv::Mat colorSupp = colorSuppression(blurred, 3);
    output_img = input_image_.clone();

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    hog.winSize = cv::Size(64, 128); // Adjust window size as needed

    Mat img_gray;
    cvtColor(colorSupp, img_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> rects;
    std::vector<double> weights;
    hog.detectMultiScale(img_gray, rects, weights, 0, cv::Size(2, 2), cv::Size(100, 100), 1.2, 0, true);

    for (size_t i = 0; i < rects.size(); i++) {
        if (weights[i] < 0.06) {
            continue;
        } 
        std::string label = "Confidence: " + std::to_string(weights[i]);
        cv::rectangle(output_img, rects[i], cv::Scalar(0, 255, 0), 2);
        cv::putText(output_img, label, cv::Point(rects[i].x, rects[i].y - 10), cv::FONT_ITALIC, 0.5, cv::Scalar(0, 255, 0), 2);
    }

}


cv::Mat playerDetection::getOutput() const {
    return output_img;
}