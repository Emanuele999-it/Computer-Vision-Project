// EMANUELE PASE 2097904

#include "imageProcessing.h"

using cv::Mat;
using std::vector;

imageProcessing::imageProcessing(const Mat & input_image){
    input_image_=input_image;
}

void imageProcessing::startprocess(){

    // --------------------- Field segmentation --------------------------
    std::cout << "Filed segmentation..." << std::endl;
    fieldSegmentation field(input_image_);
    field.startprocess();

    cv::Mat result_field_segmentation = field.returnResultImage();
    //displayMat(result_field_segmentation, "resutl field segmentation");

 
    // --------------------- Player segmentation --------------------------
    std::cout << "Player segmentation..." << std::endl;

    playerSegmentation player(input_image_);
    cv::Mat result_player_segmentation = player.startprocess();

    //displayMat(result_player_segmentation, "resutl player segmentation");

    // --------------------- Merge segmentations ------------------------
    std::cout << "Merge segmentations..." << std::endl;

    cv::Mat copy_segmentation = result_field_segmentation.clone();
    // Iterate through the source image
    for (int y = 0; y < result_player_segmentation.rows; y++) {
        for (int x = 0; x < result_player_segmentation.cols; x++) {
            // Check if the pixel is not black
            cv::Vec3b pixel = result_player_segmentation.at<cv::Vec3b>(y, x);
            if (pixel != cv::Vec3b(0, 0, 0)) {
                // Copy the pixel value to the destination image
                copy_segmentation.at<cv::Vec3b>(y, x) = pixel;
            }
        }
    }
    displayMat(copy_segmentation, "Segmentation");


    // --------------------- Player detection -----------------------------

    std::cout << "Start player detection" << std::endl;
    
    playerDetection player_detection(input_image_);
    player_detection.startprocess();

    cv:: Mat detection = player_detection.getOutput();

    displayMat(detection, "Player detection");

}


