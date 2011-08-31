#include <math.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "hybridimages.h"
#include "gaussian_kernel.h"

HybridImages::HybridImages(const std::string& input1, const std::string& input2, int left, int top, bool verbose):verbose(verbose) {
    init(cv::imread(input1), cv::imread(input2), left, top);
}

HybridImages::HybridImages(const cv::Mat& src1, const cv::Mat& src2, int left, int top, bool verbose) :verbose(verbose) {
    init(src1, src2, left, top);
}

HybridImages::~HybridImages() {}

void HybridImages::init(const cv::Mat& img1, const cv::Mat& img2, int l, int t) {
    if (verbose) {
        std::cout << "Start initialization." << std::endl;
    }
    int w1 = img1.cols, h1 = img1.rows;
    int w2 = img2.cols, h2 = img2.rows;
    
    int w, h, l1 = 0, l2 = 0, t1 = 0, t2 = 0;
    if (l > 0) {
        if (w2+l > w1) {
            w = w2+l;
        } else {
            w = w1;
        }
        l2 = l;
    } else {
        if (w2+l > w1) {
            w = w2;
        } else {
            w = w1-l;
        }
        l1 = -l;
    }
    
    if (t > 0) {
        if (h2+t > h1) {
            h = h2+t;
        } else {
            h = h1;
        }
        t2 = t;
    } else {
        if (h2+t > h1) {
            h = h2;
        } else {
            h = h1-t;
        }
        t1 = -t;
    }
    
    if (verbose) {
        std::cout << "( w,  h) = (" << w << ", " << h << ")\n"
                  << "(l1, t1) = (" << l1 << ", " << t1 << ")\n"
                  << "(l2, t2) = (" << l2 << ", " << t2 << ")" << std::endl;
    }

    dsize = cv::Size(w, h);
    ksize = cv::Size(cv::getOptimalDFTSize(w), cv::getOptimalDFTSize(h));
    
    if (verbose) {
        std::cout << "Create src1" << std::endl;
    }

    cv::Mat dst1(dsize, img1.type(), cv::Scalar::all(255));
    cv::Mat roi1 = dst1(cv::Rect(l1, t1, w1, h1));
    img1.copyTo(roi1);
    fill_external_pixel(roi1, l1, t1, w1, h1);
    dst1.convertTo(src1, CV_64FC3);
    if (verbose) {
        std::cout << "Create src2" << std::endl;
    }
    
    cv::Mat dst2(dsize, img2.type(), cv::Scalar::all(255));
    cv::Mat roi2 = dst2(cv::Rect(l2, t2, w2, h2));
    img2.copyTo(roi2);
    fill_external_pixel(roi2, l2, t2, w2, h2);
    dst2.convertTo(src2, CV_64FC3);
    if (verbose) {
        std::cout << "Finish initialization." << std::endl;
    }
}

void HybridImages::fill_external_pixel(cv::Mat& src, int l, int t, int w, int h) {
    const int width = src.cols, height = src.rows;
    for (int i = t; i > 0; i--) {
        for (int j = l; j < l+w; j++) {
            int idx = src.step*i+j*src.elemSize();
            src.data[idx+0] = src.data[idx+src.step+0];
            src.data[idx+1] = src.data[idx+src.step+1];
            src.data[idx+2] = src.data[idx+src.step+2];
        }
    }
    for (int i = t+h; i < height; i++) {
        for (int j = l; j < l+w; j++) {
            int idx = src.step*i+j*src.elemSize();
            src.data[idx+0] = src.data[idx-src.step+0];
            src.data[idx+1] = src.data[idx-src.step+1];
            src.data[idx+2] = src.data[idx-src.step+2];
        }
    }

}

// 複素数平面をdft1, dft2にコピーし，残りの行列右側部分を0で埋めた後，離散フーリエ変換を行う
void HybridImages::dft(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat(ksize, CV_64FC2);
    cv::Mat t = dst(cv::Rect(0, 0, src.cols, src.rows));
    src.copyTo(t);
    if (dst.cols > src.cols) {
        t = dst(cv::Rect(src.cols, 0, dst.cols-src.cols, src.rows));
        t = cv::Mat::zeros(t.size(), t.type());
    }
    cv::dft(dst, dst, 0, src.rows);
}

//////////////////////////////////////////
//// 1チャンネルでの Hybrid images 生成
//////////////////////////////////////////
void HybridImages::create1ch(const cv::Mat& s1, const cv::Mat& s2, const double sigma, cv::Mat& dst) {
    // 入力画像と虚数配列をマージして複素数平面を構成
    if (verbose) {
        std::cout << "Create Gaussian plane." << std::endl;
    }
    cv::Mat c1(s1.size(), CV_64FC2), c2(s1.size(), CV_64FC2);
    cv::Mat im = cv::Mat::zeros(s2.size(), CV_64FC1);
    cv::Mat mv[] = { s1, im };
    cv::merge(mv, 2, c1);
    mv[0] = s2;
    cv::merge(mv, 2, c2);
    
    if (verbose) {
        std::cout << "Discrete Fourier Transformation." << std::endl;
    }
    cv::Mat dft1, dft2;
    dft(c1, dft1);
    dft(c2, dft2);
    
    if (verbose) {
        std::cout << "Convolution in Frequency Domain." << std::endl;
    }
    // 周波数領域において，ガウシアンカーネルGを用いて以下の処理を行う
    // Output = Input1・G + Input2・(1-G)
    cv::Mat l = gs[sigma].first, h = gs[sigma].second;
    cv::multiply(dft1, l, dft1);
    cv::multiply(dft2, h, dft2);
    cv::Mat dft = dft1 + dft2;
    
    if (verbose) {
        std::cout << "Inversed Discrete Fourier Transformation." << std::endl;
    }
    // 離散フーリエ逆変換し，実数成分をdstに格納
    cv::idft(dft, dft, cv::DFT_SCALE, s1.rows);
    cv::Mat tmp = dft(cv::Rect(0, 0, s1.cols, s1.rows));
    tmp.copyTo(c1);
    cv::split(c1, mv);
    dst = mv[0];
}

//////////////////////////////
////  Hybrid images 生成
//////////////////////////////
cv::Mat& HybridImages::getHybridImages(const double sigma) {
    hi_map::iterator d = dsts.find(sigma);
    if (d != dsts.end()) {
        return d->second;
    }
    
    g_pair_map::iterator g = gs.find(sigma);
    if (g == gs.end()) {
        cv::Mat l, h;
        if (verbose) {
            std::cout << "Create Gaussian Kernel." << std::endl;
        }
        gaussian_kernel(l, h, ksize, sigma);
        gs[sigma] = std::make_pair(l, h);
    }
    
    //各チャンネルごとに処理
    std::vector<cv::Mat> mv1;
    cv::split(src1, mv1);
    std::vector<cv::Mat> mv2;
    cv::split(src2, mv2);
    std::vector<cv::Mat> mv(3);
    for (int i = 0; i < 3; i++) {
        create1ch(mv1[i], mv2[i], sigma, mv[i]);
    }
    cv::Mat dst;
    cv::merge(mv, dst);
    cv::Mat result(dsize, CV_8UC3);
    cv::convertScaleAbs(dst, result);
    dsts[sigma] = result;
    return dsts[sigma];
}
