#include "customHoughDetector.h"

using cv::Mat;

customHoughDetector::customHoughDetector(cv::Mat input_image){
    input_image_ = input_image;
}

void customHoughDetector::startprocess()
{

    // Reduce noise to input image + maintain edges
    cv::Mat result;

    cv::bilateralFilter(input_image_,result,21,10,10);
    //cv::GaussianBlur(input_image_,result, cv::Size(7,7),10,10);

    cv::Mat colorSuppressed = colorSoppression(result);
    
    displayMat(colorSuppressed, "quantized_image");

    // Convert image to grayscale
    Mat gray_img;
    cvtColor(colorSuppressed, gray_img, cv::COLOR_BGR2GRAY);

    // Get the edges chosen through the previous task
    Mat edges_img;
    Canny(gray_img, edges_img, 50, 50);

    displayMat(edges_img, "canny");

    // Set threshold max value
    const int kMaxHoughThreshold = 300;

    // Hough lines detector

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges_img, lines, 1, CV_PI / 180, 150, 0, 0);

    // Draw the lines
    Mat res_img = input_image_.clone();

    // minimum lenght of the line
    int minLineLength = std::min(input_image_.size[0], input_image_.size[1]) / 5;

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
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

    result_img_ = res_img;

    displayMat(res_img, "result img");
}

// allow to reduce the number of colors
cv::Mat customHoughDetector::colorSoppression(cv::Mat img){
    
    cv::Mat reshaped_image = img.reshape(1, img.rows * img.cols); // Reshape

    int k = 3; // Number of colors after quantization
    cv::Mat labels, centers;
    cv::Mat reshaped_image_float;
    reshaped_image.convertTo(reshaped_image_float, CV_32F); // Convert to CV_32F

    cv::kmeans(reshaped_image_float, k, labels, cv::TermCriteria(), 10, cv::KMEANS_RANDOM_CENTERS, centers); // Apply k-means

    cv::Mat quantized_image(img.size(), img.type());
    for (int i = 0; i < reshaped_image.rows; ++i) {
        quantized_image.at<cv::Vec3b>(i) = centers.at<cv::Vec3f>(labels.at<int>(i));
    }

    return quantized_image;
}

