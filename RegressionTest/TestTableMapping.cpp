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

void TestTableMapping_1() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "TABLE MAPPING REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    std::string strDataPath = "./Data/TableMappingTest_1/";
    const int TOTAL_FRAME = 9;
    const int DATA_START_ROW = 1;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBezierRank = 11;

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

void TestTableMapping_2() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "TABLE MAPPING REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    std::string strDataPath = "./data/TableMappingTest_2/";
    const int TOTAL_FRAME = 1;
    const int DATA_START_ROW = 1;
    const int BEZIER_RANK = 5;

    PR_TABLE_MAPPING_CMD stCmd;
    PR_TABLE_MAPPING_RPY stRpy;
    stCmd.nBezierRank = BEZIER_RANK;
    stCmd.fBoardPointDist = 16;

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
    if (VisionStatus::OK != stRpy.enStatus)
        return;

    std::cout << "XOffsetParam" << std::endl;
    printfMat<float>(stRpy.matXOffsetParam, 3);
    std::cout << std::endl;

    std::cout << "YOffsetParam" << std::endl;
    printfMat<float>(stRpy.matYOffsetParam, 3);

    PR_CALC_TABLE_OFFSET_CMD stCalcOffsetCmd;
    PR_CALC_TABLE_OFFSET_RPY stCalcOffsetRpy;

    stCalcOffsetCmd.nBezierRank = BEZIER_RANK;
    stCalcOffsetCmd.ptTablePos = cv::Point2f(-205, 103);
    stCalcOffsetCmd.matXOffsetParam = stRpy.matXOffsetParam;
    stCalcOffsetCmd.matYOffsetParam = stRpy.matYOffsetParam;
    stCalcOffsetCmd.fMinX = stRpy.fMinX;
    stCalcOffsetCmd.fMaxX = stRpy.fMaxX;
    stCalcOffsetCmd.fMinY = stRpy.fMinY;
    stCalcOffsetCmd.fMaxY = stRpy.fMaxY;

    PR_CalcTableOffset(&stCalcOffsetCmd, &stCalcOffsetRpy);

    std::cout << "PR_CalcTableOffset status " << ToInt32(stCalcOffsetRpy.enStatus) << std::endl;
    std::cout << "Offset " << stCalcOffsetRpy.fOffsetX << ", " << stCalcOffsetRpy.fOffsetY <<std::endl;
}

void TestTableMapping() {
    TestTableMapping_1();
    TestTableMapping_2();
}

}
}