#include <math.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "gaussian_kernel.h"

/////////////////////////
////  パラメータ類   
/////////////////////////
#define SIGMA 7.5            // ガウシアンカーネルの標準偏差パラメータ．これが大きいとINPUT1がより強く見える
#define SIDE_MAX 400        // INPUT1のどちらか1辺のサイズがこれより大きければリサイズ

// 複素数平面をdft1, dft2にコピーし，残りの行列右側部分を0で埋めた後，離散フーリエ変換を行う
void dft(const cv::Mat& src, const cv::Size ksize, cv::Mat& dst) {
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
void HybridImages1ch(const cv::Mat& s1, const cv::Mat& s2, cv::Size& ksize, const cv::Mat& l, const cv::Mat& h, cv::Mat& dst) {
    // 入力画像と虚数配列をマージして複素数平面を構成
    cv::Mat c1(s1.size(), CV_64FC2), c2(s1.size(), CV_64FC2);
    cv::Mat im = cv::Mat::zeros(s2.size(), CV_64FC1);
    cv::Mat mv[] = { s1, im };
    cv::merge(mv, 2, c1);
    mv[0] = s2;
    cv::merge(mv, 2, c2);
    
    cv::Mat dft1, dft2;
    dft(c1, ksize, dft1);
    dft(c2, ksize, dft2);
    
    // 周波数領域において，ガウシアンカーネルGを用いて以下の処理を行う
    // Output = Input1・G + Input2・(1-G)
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
void HybridImages(const cv::Mat& src1, const cv::Mat& src2, const double sig, cv::Mat& dst) {
    cv::Size ksize(cv::getOptimalDFTSize(src1.cols), cv::getOptimalDFTSize(src1.rows));
    
    cv::Mat g1(ksize, CV_64FC2), g2(ksize, CV_64FC2);
    gaussian_kernel(g1, g2, ksize, sig);
    
    //各チャンネルごとに処理
    std::vector<cv::Mat> mv1;
    cv::split(src1, mv1);
    std::vector<cv::Mat> mv2;
    cv::split(src2, mv2);
    std::vector<cv::Mat> mv(3);
    for (int i = 0; i < 3; i++) {
        HybridImages1ch(mv1[i], mv2[i], ksize, g1, g2, mv[i]);
    }
    cv::merge(mv, dst);
}

/////////////////////////////////////////
//// 画像を読み込んでHybrid imagesを保存
/////////////////////////////////////////
int HybridImages_main(const std::string& fname1, const std::string& fname2, const std::string& outname, const double sig) {
    int width, height;
    
    //INPUT1
    cv::Mat img1 = cv::imread(fname1);
    if (img1.data == NULL) {
        return -1;
    }
    if (img1.cols > img1.rows) {
        if (img1.cols > SIDE_MAX) {
            width = SIDE_MAX;
            height = SIDE_MAX * img1.rows / img1.cols;
        } else {
            width = img1.cols;
            height = img1.rows;		
        }
    } else {
        if (img1.rows > SIDE_MAX) {
            height = SIDE_MAX;
            width = SIDE_MAX * img1.cols / img1.rows;
        } else {
            width = img1.cols;
            height = img1.rows;		
        }
    }
    cv::Size s(width, height);
    cv::Mat dst1;
    cv::resize(img1, dst1, s, CV_INTER_CUBIC);
    cv::Mat src1;
    dst1.convertTo(src1, CV_64FC3);
    
    //INPUT2
    cv::Mat img2 = cv::imread(fname2);
    if (img2.data == NULL) {
        return -1;
    }
    cv::Mat dst2;
    cv::resize(img2, dst2, s, CV_INTER_CUBIC);
    cv::Mat src2;
    dst2.convertTo(src2, CV_64FC3);

    //Hybrid images 生成処理を呼び出し，結果画像を保存
    cv::Mat dst;
    HybridImages(src1, src2, sig, dst);
    cv::Mat result(cv::Size(width, height), CV_8UC3);
    cv::convertScaleAbs(dst, result);
    if (!cv::imwrite(outname, result)) {
        return -1;
    }
    
    cv::namedWindow("hybrid image", CV_WINDOW_AUTOSIZE);
    cv::imshow("hybrid image", result);
    cv::waitKey(-1);
    
    return 1;
}
