#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>

namespace AOI
{
namespace Vision
{

void TestRunLogcase()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "LOGCASE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    VisionStatus enStatus;
    enStatus = PR_RunLogCase(".\\Logcase\\LrnTmpl_Success");
    std::cout << "Run logcase result " << static_cast<int> ( enStatus ) << std::endl;

    enStatus = PR_RunLogCase(".\\Logcase\\LrnTmpl_Success\\");
    std::cout << "Run logcase result " << static_cast<int> ( enStatus ) << std::endl;

    enStatus = PR_RunLogCase("./Logcase/LrnTmpl_Success");
    std::cout << "Run logcase result " << static_cast<int> ( enStatus ) << std::endl;

    enStatus = PR_RunLogCase("./Logcase/LrnTmpl_Success/");
    std::cout << "Run logcase result " << static_cast<int> ( enStatus ) << std::endl;
}

}
}