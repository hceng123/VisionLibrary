#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include "boost/filesystem.hpp"

namespace bfs = boost::filesystem;

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

    std::string strInputLogCase(".\\Logcase\\AutoThreshold_Success.logcase");
    std::string strCloneLogCase(".\\Logcase\\AutoThreshold_Clone.logcase");
    bfs::copy_file(strInputLogCase, strCloneLogCase);

    VisionStatus enStatus = VisionStatus::OK;
    enStatus = PR_RunLogCase(".\\Logcase\\AutoThreshold_Clone.logcase");
    std::cout << "Run logcase result " << static_cast<int> (enStatus) << std::endl;

    enStatus = PR_RunLogCase("./Logcase/AutoThreshold_Clone.logcase");
    std::cout << "Run logcase result " << static_cast<int> (enStatus) << std::endl;

    bfs::remove(strCloneLogCase);
}

}
}