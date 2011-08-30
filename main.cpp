#include "hybridimages.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cmath>
#include <fstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    
    po::options_description general("Command Line Options");
    general.add_options()
        ("help,h", "show help message")
        ("show", "show image using highgui")
        ("verbose,v", "show verbose message")
        ("debug,d", "show debug infomation using highgui");
    
    std::string input1;
    std::string input2;
    double sigma;
    std::string output;
    int l, t;
    
    po::options_description config("Configuration");
    config.add_options()
        ("output,o", po::value<std::string>(&output), "output hybridimages")
        ("sigma,s", po::value<double>(&sigma)->default_value(7.5), "hybrid parameter")
        ("top,t", po::value<int>(&t)->default_value(0), "top of input file2 for hybrid image")
        ("left,l", po::value<int>(&l)->default_value(0), "left of input file2 for hybrid image");
    
    po::options_description hidden("Hidden Options");
    hidden.add_options()
        ("input-file1", po::value<std::string>(&input1), "input file1")
        ("input-file2", po::value<std::string>(&input2), "input file2");
    
    po::options_description cmdline;
    cmdline.add(general).add(config).add(hidden);
    
    po::options_description conf_opt;
    conf_opt.add(config).add(hidden);
    
    po::options_description visible;
    visible.add(general).add(config);
    
    po::positional_options_description p_opt;
    p_opt.add("input-file1", 1);
    p_opt.add("input-file2", 1);
    
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(cmdline).positional(p_opt).run(), vm);
    po::notify(vm);
    
    std::ifstream ifs("hybridimages.conf");
    po::store(po::parse_config_file(ifs, conf_opt), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
        std::cerr << "Usage:" << argv[0] << " [options] input-file1 input-file2\n"
                  << visible << std::endl;
        return 0;
    }
    
    if (vm.count("input-file1") == 0 || vm.count("input-file2") == 0) {
        std::cerr << "must input 2 image files\n"
                  << "Usage:" << argv[0] << " [options] input-file1 input-file2\n"
                  << visible << std::endl;
        return 0;
    }
    
    cv::Mat src1 = cv::imread(input1);
    cv::Mat src2 = cv::imread(input2);
    
    if (vm.count("debug")) {
        cv::namedWindow("source1", CV_WINDOW_AUTOSIZE);
        cv::imshow("source1", src1);
        cv::namedWindow("source2", CV_WINDOW_AUTOSIZE);
        cv::imshow("source2", src2);
        cv::waitKey(-1);
    }
    
    HybridImages hi(src1, src2, l, t, vm.count("verbose"));
    cv::Mat result = hi.getHybridImages(sigma);
    
    if (vm.count("output") && !cv::imwrite(output, result)) {
        return EXIT_FAILURE;
    }
    
    if (vm.count("show")) {
        cv::namedWindow("hybrid image", CV_WINDOW_AUTOSIZE);
        cv::imshow("hybrid image", result);
        cv::waitKey(-1);
    }
    
    return 0;
}
