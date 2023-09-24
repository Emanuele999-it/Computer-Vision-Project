// EMANUELE PASE 2097904

#ifndef PLAYER_TEAM_H
#define PLAYER_TEAM_H

#include <opencv2/opencv.hpp>

#include <iostream>

// Allow to visualize images with an inline function
const std::vector<int> getPlayerTeam(const cv::Mat &img, const std::vector<cv::Rect> *rectangles);
	

#endif // PLAYER_TEAM_H