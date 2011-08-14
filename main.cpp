#include "hybridimages.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <boost/lexical_cast.hpp>

int main(int argc, char* argv[]){
    std::string input1 = argv[1];
    std::string input2 = argv[2];
    double sigma = boost::lexical_cast<double>(argv[3]);
    
    HybridImages hi(input1, input2);
    cv::Mat result = hi.getHybridImages(sigma);
    
    cv::namedWindow("hybrid image", CV_WINDOW_AUTOSIZE);
    cv::imshow("hybrid image", result);
    cv::waitKey(-1);
    
    return 1;
}
