#include <iostream>
#include <iomanip>
#include "UtilityFunc.h"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"

namespace AOI
{
namespace Vision
{

static void PrintRpy(const PR_TABLE_MAPPING_RPY &stRpy) {
    std::cout << "Table mapping status " << ToInt32(stRpy.enStatus) << std::endl;
    if (VisionStatus::OK != stRpy.enStatus)
        return;
    std::cout << "XOffsetParam" << std::endl;
    printfMat<float>(stRpy.matXOffsetParam, 3);
    std::cout << std::endl;

    std::cout << "YOffsetParam" << std::endl;
    printfMat<float>(stRpy.matYOffsetParam, 3);
}

void TestTableMapping() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "TABLE MAPPING REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    std::string strDataPath = "./Data/TableMappingMergeTest/";
    const int TOTAL_FRAME = 9;
    const int DATA_START_ROW = 1;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBazierRank = 11;

    for (int i = 1; i <= TOTAL_FRAME; ++ i) {
        char chArrFileName[100];
        _snprintf(chArrFileName, sizeof(chArrFileName), "%02d.csv", i); 
        auto vecOfVecFloat = readDataFromFile(strDataPath + chArrFileName);

        PR_TABLE_MAPPING_CMD::VectorOfFramePoint vecFramePoint;
        vecFramePoint.reserve(vecOfVecFloat.size());

        for (int row = DATA_START_ROW; row < vecOfVecFloat.size(); ++ row) {
            PR_TABLE_MAPPING_CMD::FramePoint framePoint;
            framePoint.targetPoint = cv::Point2f(vecOfVecFloat[row][1], vecOfVecFloat[row][2]);
            framePoint.actualPoint = cv::Point2f(vecOfVecFloat[row][3], vecOfVecFloat[row][4]);
            vecFramePoint.push_back(framePoint);
        }

        stCmd.vecFramePoints.push_back(vecFramePoint);
    }

    PR_TableMapping(&stCmd, &stRpy);
    PrintRpy(stRpy);
}

}
}