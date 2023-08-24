#include <iostream>
#include <opencv2/opencv.hpp>
#include "imageProcessing.h"

using std::endl;
using std::cerr;

int main (int argc, char** argv){

    // Check if path to image is provided correctly as command line argument
	if (argc != 2)
	{
			cerr << "Usage: " << argv[0] << " <filename>" << endl;
			return -1;
	}

    // Load the image
    cv::Mat image = cv::imread(argv[1]);

    // Check if image is loaded successfully
    if (image.empty())
    {
            cerr << "Error: Could not load image " << argv[1] << endl;
            return -1;
    }

    // Start image processing
    imageProcessing imgProc(image);
    imgProc.startprocess();
}