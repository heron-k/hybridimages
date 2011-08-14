#include <math.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "hybridimages.h"
#include "gaussian_kernel.h"

HybridImages::HybridImages(const std::string& input1, const std::string& input2) {
    init(cv::imread(input1), cv::imread(input2));
}

HybridImages::HybridImages(const cv::Mat& src1, const cv::Mat& src2) {
    init(src1, src2);
}

void HybridImages::init(const cv::Mat& img1, const cv::Mat& img2) {
    int width, height;
    if (img1.cols > img1.rows) {
        if (img1.cols > MaxWidth) {
            width = MaxWidth;
            height = MaxWidth * img1.rows / img1.cols;
        } else {
            width = img1.cols;
            height = img1.rows;		
        }
    } else {
        if (img1.rows > MaxWidth) {
            height = MaxWidth;
            width = MaxWidth * img1.cols / img1.rows;
        } else {
            width = img1.cols;
            height = img1.rows;		
        }
    }
    dsize = cv::Size(width, height);
    ksize = cv::Size(cv::getOptimalDFTSize(width), cv::getOptimalDFTSize(height));
    cv::Mat dst1;
    cv::resize(img1, dst1, dsize, CV_INTER_CUBIC);
    dst1.convertTo(src1, CV_64FC3);
    
    cv::Mat dst2;
    cv::resize(img2, dst2, dsize, CV_INTER_CUBIC);
    dst2.convertTo(src2, CV_64FC3);
    std::cerr << "initialized" << std::endl;
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
    std::cerr << "DFT" << std::endl;
}

//////////////////////////////////////////
//// 1チャンネルでの Hybrid images 生成
//////////////////////////////////////////
void HybridImages::create1ch(const cv::Mat& s1, const cv::Mat& s2, const double sigma, cv::Mat& dst) {
    // 入力画像と虚数配列をマージして複素数平面を構成
    cv::Mat c1(s1.size(), CV_64FC2), c2(s1.size(), CV_64FC2);
    cv::Mat im = cv::Mat::zeros(s2.size(), CV_64FC1);
    cv::Mat mv[] = { s1, im };
    cv::merge(mv, 2, c1);
    mv[0] = s2;
    cv::merge(mv, 2, c2);
    
    cv::Mat dft1, dft2;
    dft(c1, dft1);
    dft(c2, dft2);
    
    // 周波数領域において，ガウシアンカーネルGを用いて以下の処理を行う
    // Output = Input1・G + Input2・(1-G)
    cv::Mat l = gs[sigma].first, h = gs[sigma].second;
    cv::multiply(dft1, l, dft1);
    cv::multiply(dft2, h, dft2);
    cv::Mat dft = dft1 + dft2;
    
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
