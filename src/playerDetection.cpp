#include "playerDetection.h"

using cv::Mat;

playerDetection::playerDetection(Mat input_image){
    input_image_ = input_image;

}

void playerDetection::startprocess(){
/*
    cv::Mat input_clone = input_image_.clone();

    Mat blurred;
    cv::bilateralFilter(input_clone, blurred, 3, 40, 11);
    //cv::GaussianBlur(blurred,blurred, cv::Size(3,3),2);

    //cv::Mat colorSupp = colorSuppression(blurred, 4);
    output_img = input_image_.clone();

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    Mat img_gray;
    cvtColor(blurred, img_gray, cv::COLOR_BGR2GRAY);

    displayMat(img_gray, "img_gray",1);

    // Apply local contrast enhancement to the grayscale image using CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(15); // You can adjust this value to control the enhancement
    clahe->apply(img_gray, img_gray);
    
    cv::equalizeHist(img_gray, img_gray);

    displayMat(img_gray, "img_gray eq",1);

    //cv::threshold(img_gray, img_gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<cv::Rect> rects;
    std::vector<double> weights;
    hog.detectMultiScale(img_gray, rects, weights, 0, cv::Size(4, 4), cv::Size(), 1.1, 1, false);


    for (size_t i = 0; i < rects.size(); i++) {
        //if (weights[i] < 0.05) {
        //    continue;
        //} 
        std::string label = "Confidence: " + std::to_string(weights[i]);
        cv::rectangle(output_img, rects[i], cv::Scalar(0, 255, 0), 2);
        cv::putText(output_img, label, cv::Point(rects[i].x, rects[i].y - 10), cv::FONT_ITALIC, 0.5, cv::Scalar(0, 255, 0), 2);
    }

*/

    cv::Mat input_clone = input_image_.clone();

    Mat blurred;
    cv::bilateralFilter(input_clone, blurred, 7, 1000, 400);
    cv::GaussianBlur(blurred,blurred, cv::Size(5,5), 0.5);

    //cv::Mat colorSupp = colorSuppression(blurred, 5);
    output_img = input_image_.clone();

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());


    hog.gammaCorrection = false;
    hog.L2HysThreshold = 230;
    hog.nlevels = 128;

    Mat img_gray;
    cvtColor(blurred, img_gray, cv::COLOR_BGR2GRAY);

    

    // Apply local contrast enhancement to the grayscale image using CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4); // You can adjust this value to control the enhancement
    clahe->apply(img_gray, img_gray);

    cv::equalizeHist(img_gray,img_gray);
    //displayMat(img_gray, "imgray",1);
    //cv::GaussianBlur(img_gray,img_gray, cv::Size(3,3),0.5,3);
    
    //cv::threshold(img_gray, img_gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    int exp = 6;
    int winSizeX = pow(2, exp);
    int winSizeY = pow(2, exp + 1);

    std::vector<cv::Rect> all_rects;

    // Ensure that winSizeX and winSizeY are multiples of blockStride
    if (winSizeX <= img_gray.cols && winSizeY <= img_gray.rows) {
        hog.winSize = cv::Size(winSizeX, winSizeY);
        std::vector<cv::Rect> rects;
        std::vector<double> weights;
/*
        hog.detectMultiScale(img_gray, rects, weights, 0, cv::Size(2, 2), cv::Size(), 1.04, 4, false);

        for (size_t i = 0; i < rects.size(); i++) {
            //if (weights[i] < 0.05) {
            //    continue;
            //}
            //std::string label = "Confidence: " + std::to_string(weights[i]);
            cv::rectangle(output_img, rects[i], cv::Scalar(0, 255, 0), 2);
            //cv::putText(output_img, label, cv::Point(rects[i].x, rects[i].y - 10), cv::FONT_ITALIC, 0.5, cv::Scalar(0, 255, 0), 2);

        }
*/
        for (float scale = 1.1; scale >= 0.1; scale -= 0.2){

            cv::Mat resized_img;
            cv::resize(img_gray, resized_img, cv::Size(), scale, scale);

            hog.detectMultiScale(resized_img, rects, weights, 0, cv::Size(2, 2), cv::Size(15,15), 1.04, 1, false);

            for(cv::Rect element : rects){
                // Adjust the rectangles to the original image size
                element.x /= scale;
                element.y /= scale;
                element.width /= scale;
                element.height /= scale;

                all_rects.push_back(element);
            }
/*
            // Adjust the rectangles to the original image size and draw them
            for (size_t i = 0; i < rects.size(); i++){

                rects[i] = cv::Rect(rects[i].x / scale, rects[i].y / scale, rects[i].width / scale, rects[i].height / scale);
                cv::rectangle(output_img, rects[i], cv::Scalar(0, 255, 0), 2);
            }
*/
        }
    }

    std::cout << "all_rects numbers: " << all_rects.size() << std::endl;

    std::vector<cv::Rect> filteredDetections;

    for (size_t i = 0; i < all_rects.size(); i++) {
        bool keep = true;

        for (size_t j = 0; j < all_rects.size(); j++) {
            // Define your similarity criteria here.
            // For example, you can check if the two rectangles have a significant overlap.
            if ( i != j){
                float overlap = (float)(all_rects[i] & all_rects[j]).area() / (float)(all_rects[i] | all_rects[j]).area();

                if (overlap > 0.99 || all_rects[i].x >= all_rects[j].x &&
                    all_rects[i].y >= all_rects[j].y &&
                    (all_rects[i].x + all_rects[i].width) <= (all_rects[j].x + all_rects[j].width) &&
                    (all_rects[i].y + all_rects[i].height) <= (all_rects[j].y + all_rects[j].height)) {

                    keep = false; // Discard one of the rectangles if they are too similar or one is inside the other
                    break;
                }
            }
        }

        if (keep) {
            filteredDetections.push_back(all_rects[i]);
        }
    }

        std::cout << "filteredDetections numbers: " << filteredDetections.size() << std::endl;


    for (size_t i = 0; i < filteredDetections.size(); i++){
        filteredDetections[i] = cv::Rect(filteredDetections[i].x, filteredDetections[i].y, filteredDetections[i].width, filteredDetections[i].height);
        cv::rectangle(output_img, filteredDetections[i], cv::Scalar(0, 255, 0), 2);
    }
}


cv::Mat playerDetection::getOutput() const {
    return output_img;
}