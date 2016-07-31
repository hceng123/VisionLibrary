#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include "../VisionLibrary/VisionAPI.h"

using namespace AOI::Vision;

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
}