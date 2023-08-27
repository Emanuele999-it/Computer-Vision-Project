#include "customHoughDetector.h"

using cv::Mat;

customHoughDetector::customHoughDetector(cv::Mat input_image){
    input_image_ = input_image;
}

void customHoughDetector::startprocess()
{

    // Reduce noise to input image + maintain edges
    cv::Mat resultBilateral, resultGaussian, blurred_input;

    cv::bilateralFilter(input_image_,resultBilateral,21,21,21);
    cv::GaussianBlur(resultBilateral,resultGaussian, cv::Size(3,3),1.5,1.5);

    cv::Mat colorSuppressed = colorSuppression(resultGaussian);

    displayMat(colorSuppressed, "quantized_image");

    // Convert image to grayscale
    Mat gray_img;
    cv::bilateralFilter(input_image_,blurred_input,21,21,21);
    cvtColor(blurred_input, gray_img, cv::COLOR_BGR2GRAY);

    // Get the edges chosen through the previous task
    Mat edges_img;
    Canny(gray_img, edges_img, 150, 200);

    displayMat(edges_img, "canny");

    // ------------------------------ Contours filtering -------------------------------------------

    // Find contours in the edge image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filter out complex contours based on area
    std::vector<std::vector<cv::Point>> simpleContours;
    for (const auto& contour : contours) {
        double contourArea = cv::contourArea(contour);
        if (contourArea < 35) {
            simpleContours.push_back(contour);
        }
    }

    // Create a binary image from the filtered contours
    cv::Mat binaryImage = cv::Mat::zeros(input_image_.size(), CV_8UC1);
    cv::drawContours(binaryImage, simpleContours, -1, cv::Scalar(255), cv::FILLED);

    std::cout << "prima del disastro";

    displayMat(binaryImage, "binaryImage");

    // -------------------------------------------------------------------------

    

    // Hough lines detector

    Mat Hough_img = houg(binaryImage);

    displayMat(Hough_img, "result img");

}

// allow to reduce the number of colors
cv::Mat customHoughDetector::colorSuppression(cv::Mat img){
    
    cv::Mat reshaped_image = img.reshape(1, img.rows * img.cols); 

    int k = 3; // Number of colors after quantization
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



Mat customHoughDetector::houg(Mat img){

    std::vector<cv::Vec2f> lines;
    //cv::HoughLinesP(edges_img, lines, 1, CV_PI / 180, 150, 0, 0);
    cv::HoughLines (img, lines, 1, CV_PI / 180, 100, 0, 0);


    // Filter out similar lines based on position and orientation
    std::vector<cv::Vec2f> filteredLines;
    double minPositionThreshold = 10.0; 
    double minOrientationThreshold = CV_PI / 36;

    for (const cv::Vec2f& line : lines) {
        bool similarLineFound = false;

        for (const cv::Vec2f& filteredLine : filteredLines) {
            double positionDifference = std::abs(line[0] - filteredLine[0]);
            double orientationDifference = std::abs(line[1] - filteredLine[1]);

            if (positionDifference < minPositionThreshold &&
                orientationDifference < minOrientationThreshold) {
                similarLineFound = true;
                break;
            }
        }

        if (!similarLineFound) {
            filteredLines.push_back(line);
        }
    }

    

    // Draw the lines
    Mat res_img = input_image_.clone();

    // Minimum lenght of the line
    int minLineLength = std::min(input_image_.size[0], input_image_.size[1]) / 5;

    // Display lines
    for (size_t i = 0; i < filteredLines.size(); i++)
    {
        float rho = filteredLines[i][0];
        float theta = filteredLines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;

        // Given the point (x0, y0) and the coefficients a and b (representing theta)
        // I need to evaluate two points pt1 and pt2 to draw the line
        const int kLineOffset = 1000;
        
        cv::Point pt1(cvRound(x0 + kLineOffset * (-b)), cvRound(y0 + kLineOffset * (a)));
        cv::Point pt2(cvRound(x0 - kLineOffset * (-b)), cvRound(y0 - kLineOffset * (a)));
        
        // Calculate line length and filter based on the threshold
        double lineLength = cv::norm(pt1 - pt2);
        if (lineLength >= minLineLength) {
            cv::line(res_img, pt1, pt2, cv::Scalar(0, 0, 255), 2);
        }
        
    }

    return res_img;
}

