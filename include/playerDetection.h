#ifndef PLAYER_DETECTION_H
#define PLAYER_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "displayMat.h"
#include "colorSuppression.h"


class playerDetection{

    private:
        cv::Mat input_image_;
        cv::Mat output_img;
        //void ResizeBoxes(cv::Rect& box);

    public:
        playerDetection(cv::Mat input_image);
        void startprocess();
        cv::Mat getOutput() const;
        
};

#endif // PLAYER_DETECTION_H