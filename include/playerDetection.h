// EMANUELE PASE 2097904

#ifndef PLAYER_DETECTION_H
#define PLAYER_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <cmath>

#include "displayMat.h"
#include "colorSuppression.h"
#include "playerTeam.h"


class playerDetection{

    private:
        cv::Mat input_image_;
        cv::Mat output_img;
        //void ResizeBoxes(cv::Rect& box);

    public:
        playerDetection(const cv::Mat & input_image);
        void startprocess();
        cv::Mat getOutput() const;
        
};

#endif // PLAYER_DETECTION_H