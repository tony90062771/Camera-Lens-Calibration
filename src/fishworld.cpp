#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <glob.h>
#include <sys/stat.h>

// 相機參數
const cv::Size DIM(1280, 720);
const cv::Mat K = (cv::Mat_<double>(3, 3) << 315.601188269323, 0                , 643.2953923945038,
                                             0               , 315.1159925901694, 355.1132032478912,
                                             0               , 0                , 1);

const cv::Mat D = (cv::Mat_<double>(1, 4) << 0.1892211046287096, -0.04989273049545627, -0.001938553334422876, 0.002080231663473995);


void undistortFisheyePoints(const std::vector<cv::Point2f> &distortedPoints, std::vector<cv::Point2f> &undistortedPoints){
    // 校正像素座標
    cv::fisheye::undistortPoints(distortedPoints, undistortedPoints, K, D);
}
void distortFisheyePoints(const std::vector<cv::Point2f> &undistortedPoints, std::vector<cv::Point2f> &distortedPoints){
    // 將校正後的像素座標轉換回魚眼像素座標
    cv::fisheye::distortPoints(undistortedPoints, distortedPoints, K, D);
}
// 將魚眼像素座標轉成世界座標的函式
cv::Point2f undistortedToWorld(const cv::Point2f &fisheyePoint, const cv::Mat &homographyMatrix){
    // 校正像素座標
    std::vector<cv::Point2f> distortedPoints = {fisheyePoint};
    std::vector<cv::Point2f> undistortedPoints;
    undistortFisheyePoints(distortedPoints, undistortedPoints);

    // 取校正後的座標點
    cv::Point2f undistortedPoint = undistortedPoints[0];

    // 將點轉換為齊次坐標
    cv::Mat pointMat = (cv::Mat_<double>(3, 1) << undistortedPoint.x, undistortedPoint.y, 1.0);
    // 應用單應性矩陣
    cv::Mat worldMat = homographyMatrix * pointMat;
    // 轉換回非齊次坐標
    return cv::Point2f(worldMat.at<double>(0, 0) / worldMat.at<double>(2, 0),
                       worldMat.at<double>(1, 0) / worldMat.at<double>(2, 0));
}

// 將世界坐標轉成像素座標的函式
cv::Point2f worldToFisheye(const cv::Point2f &worldPoint, const cv::Mat &homographyMatrix){
    // 將世界座標轉換為齊次坐標
    cv::Mat pointMat = (cv::Mat_<double>(3, 1) << worldPoint.x, worldPoint.y, 1.0);
    // 應用逆單應性矩陣
    cv::Mat invHomographyMatrix = homographyMatrix.inv();
    cv::Mat imageMat = invHomographyMatrix * pointMat;
    // 轉換回非齊次坐標
    return cv::Point2f(imageMat.at<double>(0, 0) / imageMat.at<double>(2, 0),
                       imageMat.at<double>(1, 0) / imageMat.at<double>(2, 0));


    /* 將校正後的圖像坐標轉換回魚眼像素座標
    std::vector<cv::Point2f> undistortedPoints = {undistortedPoint};
    std::vector<cv::Point2f> distortedPoints;
    distortFisheyePoints(undistortedPoints, distortedPoints);

    return distortedPoints[0];*/
}
// 定義全局變數來存儲單應性矩陣
cv::Mat homographyMatrix_front;
cv::Mat homographyMatrix_back;
cv::Mat homographyMatrix_left;
cv::Mat homographyMatrix_right;
//單應性矩陣副涵式
void gethomographyMatrix(){ 
    //front
    // 世界坐標點
    std::vector<cv::Point2f> H_worldPoint_front;
    H_worldPoint_front.push_back(cv::Point2f(-300, 0));
    H_worldPoint_front.push_back(cv::Point2f(-300, 500));
    H_worldPoint_front.push_back(cv::Point2f(-300, 1000));
    H_worldPoint_front.push_back(cv::Point2f(300, 1000));
    H_worldPoint_front.push_back(cv::Point2f(300, 500));
    H_worldPoint_front.push_back(cv::Point2f(300, 0));
    // 像素座標點
    std::vector<cv::Point2f> H_fishPoint_front;
    H_fishPoint_front.push_back(cv::Point2f(114, 425));
    H_fishPoint_front.push_back(cv::Point2f(435, 201));
    H_fishPoint_front.push_back(cv::Point2f(523, 171));
    H_fishPoint_front.push_back(cv::Point2f(722, 170));
    H_fishPoint_front.push_back(cv::Point2f(805, 198));
    H_fishPoint_front.push_back(cv::Point2f(1128, 403));
    homographyMatrix_front = cv::findHomography(H_fishPoint_front, H_worldPoint_front);
}

// 去畸變影像函式
cv::Mat distortImage(const std::string &imgPath) {
    cv::Mat img = cv::imread(imgPath);
    cv::Mat map1, map2;
    cv::Mat undistortedImg;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_64F), K, DIM, CV_16SC2, map1, map2);
    cv::remap(img, undistortedImg, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    return  undistortedImg;
}


int main() {
    cv::Mat image_front=distortImage("../input/front.jpg");

    gethomographyMatrix();
    //cv::namedWindow("test",cv::WINDOW_NORMAL);

    //front
    for(int i=-300;i<1501;i=i+50){
        for(int j=-300;j<301;j=j+50){
            cv::circle(image_front, worldToFisheye(cv::Point2f(static_cast<float>(j), static_cast<float>(i)), homographyMatrix_front), 5, cv::Scalar(0, 255, 255), -1);
        }
    }
    cv::imwrite("../output/front-yp.jpg",image_front);
    

    //cv::imshow("test",image_front);
    //cv::waitKey(100000);
    cv::destroyAllWindows();

    return 0;
}
