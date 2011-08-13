#include "hybridimages.h"
#include <boost/lexical_cast.hpp>

int main(int argc, char* argv[]){
    std::string input1 = argv[1];
    std::string input2 = argv[2];
    std::string output = argv[3];
    double sigma = boost::lexical_cast<double>(argv[4]);
    
    HybridImages_main(input1, input2, output, sigma);
    
    return 1;
}
