#ifndef GAUSSIAN_KERNEL_H
#define GAUSSIAN_KERNEL_H

#include <opencv/cv.h>

void gaussian_kernel(cv::Mat& g1, cv::Mat& g2, const cv::Size& ksize, const double sig);

#endif
