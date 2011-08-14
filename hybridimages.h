#ifndef HYBRIDIMAGES_H
#define HYBRIDIMAGES_H

#include <string>
#include <opencv/cv.h>
#include <tr1/unordered_map>
#include <utility>

typedef std::tr1::unordered_map<double, cv::Mat> hi_map;
typedef std::pair<cv::Mat, cv::Mat> g_pair;
typedef std::tr1::unordered_map<double, g_pair> g_pair_map;

class HybridImages {
private:
    cv::Mat src1;
    cv::Mat src2;
    cv::Size ksize;
    cv::Size dsize;
    hi_map dsts;
    g_pair_map gs;
    void dft(const cv::Mat& src, cv::Mat& dst);
    void create1ch(const cv::Mat& s1, const cv::Mat& s2, const double sigma, cv::Mat& dst);
    void init(const cv::Mat& src1, const cv::Mat& src2);
    static const int MaxWidth = 400;
public:
    HybridImages(const std::string& input1, const std::string& input2);
    HybridImages(const cv::Mat& src1, const cv::Mat& src2);
    cv::Mat& getHybridImages(const double sigma);
};

#endif
