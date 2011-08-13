////////////////////////////////////
//// ガウシアンカーネル作成
////////////////////////////////////

#include "gaussian_kernel.h"

void swap(cv::Mat& src, const double xm, const double ym) {
    const int cx = static_cast<int>(xm), cy = static_cast<int>(ym);
    cv::Mat q1, q2, q3, q4, tmp(src.rows/2, src.cols/2, CV_64FC2);
    q1 = src(cv::Rect( 0,  0, cx, cy));
    q2 = src(cv::Rect(cx,  0, cx, cy));
    q3 = src(cv::Rect(cx, cy, cx, cy));
    q4 = src(cv::Rect( 0, cy, cx, cy));
    q3.copyTo(tmp);
    q1.copyTo(q3);
    tmp.copyTo(q1);
    q4.copyTo(tmp);
    q2.copyTo(q4);
    tmp.copyTo(q2);
}

void gaussian_kernel(cv::Mat& g1, cv::Mat& g2, const cv::Size& ksize, const double sig) {
    const double ss2 = sig*sig*2;
    const int xs = ksize.width, ys = ksize.height;
    const double xm = xs/2.0, ym = ys/2.0;
    for (int y = 0; y < ys; y++) {
        double* row1 = g1.ptr<double>(y);
        double* row2 = g2.ptr<double>(y);
        for (int x = 0; x < xs; x++) {
            double gaud = (xm-x)*(xm-x)+(ym-y)*(ym-y);
            double gaui = exp(-gaud/ss2);
            row1[x*2] = gaui;
            row1[x*2+1] = gaui;
            row2[x*2] = 1-gaui;
            row2[x*2+1] = 1-gaui;
        }
    }
    swap(g1, xm, ym);
    swap(g2, xm, ym);    
}
