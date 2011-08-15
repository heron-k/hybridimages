#include "hybridimages.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <fstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    
    po::options_description general("Command Line Options");
    general.add_options()
        ("help,h", "show help message")
        ("verbose,v", "show verbose message");
    
    std::string input1;
    std::string input2;
    double sigma;
    std::string output;

    po::options_description config("Configuration");
    config.add_options()
        ("output,o", po::value<std::string>(&output), "output hybridimages")
        ("sigma,s", po::value<double>(&sigma)->default_value(7.5), "hybrid parameter");
    
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
    
    HybridImages hi(input1, input2);
    cv::Mat result = hi.getHybridImages(sigma);
    
    if (argc >= 5 && !cv::imwrite(output, result)) {
        return -1;
    }
    
    cv::namedWindow("hybrid image", CV_WINDOW_AUTOSIZE);
    cv::imshow("hybrid image", result);
    cv::waitKey(-1);
    
    return 0;
}
