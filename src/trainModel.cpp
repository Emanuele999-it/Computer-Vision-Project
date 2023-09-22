#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <filesystem>


using namespace cv;
using namespace cv::ml;
using namespace std;

namespace fs = std::filesystem;


vector<float> get_svm_detector(const Ptr<SVM> &svm);
void convert_to_ml(const std::vector<Mat> &train_samples, Mat &trainData);
void load_images(const String &dirname, vector<Mat> &img_lst,  bool pos, bool showImages);
void sample_neg(const vector<Mat> &full_neg_lst, vector<Mat> &neg_lst, const Size &size);
void computeHOGs(const Size wsize, const vector<Mat> &img_lst, vector<Mat> &gradient_lst, bool use_flip);

vector<float> get_svm_detector(const Ptr<SVM> &svm)
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);
    CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
    CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
              (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
    CV_Assert(sv.type() == CV_32F);
    vector<float> hog_detector(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}


/*
 * Convert training/testing set to be used by OpenCV Machine Learning algorithms.
 * TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
 * Transposition of samples are made if needed.
 */
void convert_to_ml(const vector<Mat> &train_samples, Mat &trainData)
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
    Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = Mat(rows, cols, CV_32FC1);
    for (size_t i = 0; i < train_samples.size(); ++i)
    {
        CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);
        if (train_samples[i].cols == 1)
        {
            transpose(train_samples[i], tmp);
            tmp.copyTo(trainData.row((int)i));
        }
        else if (train_samples[i].rows == 1)
        {
            train_samples[i].copyTo(trainData.row((int)i));
        }
    }
}


void load_images(const String &dirname, vector<Mat> &img_lst, bool pos)
{

    cv::Size newSize = cv::Size(64,128);
    vector<String> files;
    glob(dirname, files);
    cout << "Files found in directory: " << files.size() << endl;
    for (size_t i = 0; i < files.size(); ++i)
    {
        Mat img = imread(files[i]); // load the image

        if (pos)
            resize(img, img, newSize);

        if (img.empty())
        {
            cout << files[i] << " is invalid!" << endl; // invalid image, skip it.
            continue;
        }
        if (showImages)
        {
            imshow("image", img);
            waitKey(200);
        }
        img_lst.push_back(img);
    }
}

void sample_neg(const vector<Mat> &full_neg_lst, vector<Mat> &neg_lst, const Size &size)
{
    Rect box;
    box.width = size.width;
    box.height = size.height;
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < full_neg_lst.size(); i++)
        if (full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height)
        {
            box.x = rand() % (full_neg_lst[i].cols - box.width);
            box.y = rand() % (full_neg_lst[i].rows - box.height);
            Mat roi = full_neg_lst[i](box);
            neg_lst.push_back(roi.clone());
        }
}

void computeHOGs(const Size wsize, const vector<Mat> &img_lst, vector<Mat> &gradient_lst, bool use_flip)
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector<float> descriptors;
    for (size_t i = 0; i < img_lst.size(); i++)
    {
        if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height)
        {
            Rect r = Rect((img_lst[i].cols - wsize.width) / 2,
                          (img_lst[i].rows - wsize.height) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor(img_lst[i](r), gray, COLOR_BGR2GRAY);
            hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
            gradient_lst.push_back(Mat(descriptors).clone());
            if (use_flip)
            {
                flip(gray, gray, 1);
                hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));
                gradient_lst.push_back(Mat(descriptors).clone());
            }
        }
    }
}


int main(int argc, char **argv)
{
    if (argc < 3)
	{
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return -1;
	}

    String pos_dir = argv[1];
    String neg_dir = argv[2];
    int detector_width = argc >= 5 ? (int)argv[3] : 64;
    int detector_height = argc >= 5 ? (int)argv[4] : 128;


    vector<Mat> pos_lst, full_neg_lst, neg_lst, gradient_lst;
    vector<int> labels;
    
    load_images(pos_dir, pos_lst, true);

    load_images(neg_dir, neg_lst, false);

    if (pos_lst.size() == 0)
    {
        clog << "no image in " << pos_dir << endl;
        return 1;
    }

    Size pos_image_size = pos_lst[0].size();

    if (detector_width && detector_height)
    {
        pos_image_size = Size(detector_width, detector_height);
    }
    else
    {
        for (size_t i = 0; i < pos_lst.size(); ++i)
        {
            if (pos_lst[i].size() != pos_image_size)
            {
                cout << "All positive images should be same size!" << endl;
                exit(1);
            }
        }
        pos_image_size = pos_image_size / 8 * 8;
    }

    load_images(neg_dir, full_neg_lst, false);

    sample_neg(full_neg_lst, neg_lst, pos_image_size);

    computeHOGs(pos_image_size, pos_lst);

    size_t positive_count = gradient_lst.size();
    labels.assign(positive_count, +1);

    computeHOGs(pos_image_size, neg_lst);
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert(labels.end(), negative_count, -1);
    CV_Assert(positive_count < labels.size());

    Mat train_data;
    convert_to_ml(gradient_lst, train_data);

    Ptr<SVM> svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3));
    svm->setGamma(0);
    svm->setKernel(SVM::LINEAR);
    svm->setNu(0.5);
    svm->setP(0.1);             // for EPSILON_SVR, epsilon in loss function?
    svm->setC(0.01);            // From paper, soft classifier
    svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train(train_data, ROW_SAMPLE, labels);
    
    HOGDescriptor hog;
    hog.winSize = pos_image_size;
    hog.setSVMDetector(get_svm_detector(svm));
    hog.save("model.yml");
    return 0;
}
