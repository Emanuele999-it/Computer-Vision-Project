#include "imageProcessing.h"

using cv::Mat;
using std::vector;

cv::RNG rng(1000);

imageProcessing::imageProcessing(Mat input_image){
    input_image_=input_image;
}


void imageProcessing::startprocess(){


/*
    // --------------------- Field segmentation --------------------------
    fieldSegmentation field(input_image_);
    field.startprocess();

    cv::Mat result_field_segmentation = field.returnResultImage();
    displayMat(result_field_segmentation, "resutl field segmentation");
*/

/*  
    // --------------------- Player segmentation --------------------------
    playerSegmentation player(input_image_, result_field_segmentation);
    cv::Mat result_player_segmentation = player.startprocess();

    displayMat(result_player_segmentation, "resutl player segmentation");

*/

    // --------------------- Player detection -----------------------------
    playerDetection player_detection(input_image_);
    player_detection.startprocess();

    cv:: Mat detection = player_detection.getOutput();

    displayMat(detection, "player detection");
}


