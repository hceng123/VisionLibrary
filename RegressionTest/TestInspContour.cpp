#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>

namespace AOI
{
namespace Vision
{

static void PrintLrnReply(const PR_LRN_CONTOUR_RPY &stRpy) {
    std::cout << "Learn contour status: " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Result threshold: " << stRpy.nThreshold << std::endl;
    std::cout << "Number of contour: " << stRpy.vecContours.size() << std::endl;
}

static void PrintInspReply(const PR_INSP_CONTOUR_RPY &stRpy) {
    std::cout << "Inspect contour status: " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Number of defects: " << stRpy.vecDefectContour.size() << std::endl;
}

static void TestInspContour_1() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT CONTOUR REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_LRN_CONTOUR_CMD stLrnCmd;
    PR_LRN_CONTOUR_RPY stLrnRpy;
    stLrnCmd.bAutoThreshold = true;
    stLrnCmd.matInputImg = cv::imread ( "./data/RoundContourTmpl.png", cv::IMREAD_GRAYSCALE );
    stLrnCmd.rectROI = cv::Rect(0, 0, 276, 253);
    PR_LrnContour(&stLrnCmd, &stLrnRpy);
    PrintLrnReply(stLrnRpy);

    PR_INSP_CONTOUR_CMD stInspCmd;
    PR_INSP_CONTOUR_RPY stInspRpy;
    stInspCmd.matInputImg = cv::imread ( "./data/RoundContourTmpl.png", cv::IMREAD_GRAYSCALE );
    stInspCmd.rectROI = cv::Rect(0, 0, 276, 253);
    stInspCmd.nRecordId = stLrnRpy.nRecordId;
    PR_InspContour(&stInspCmd, &stInspRpy);
    PrintInspReply(stInspRpy);

    stInspCmd.matInputImg = cv::imread ( "./data/RoundContourInsp.png", cv::IMREAD_GRAYSCALE );
    stInspCmd.rectROI = cv::Rect(0, 0, 276, 253);
    stInspCmd.nRecordId = stLrnRpy.nRecordId;
    PR_InspContour(&stInspCmd, &stInspRpy);
    PrintInspReply(stInspRpy);

    PR_FreeRecord ( stLrnRpy.nRecordId );
}

static void TestInspContour_2() {
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl << "INSPECT CONTOUR REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "---------------------------------------------";
    std::cout << std::endl;

    PR_LRN_CONTOUR_CMD stLrnCmd;
    PR_LRN_CONTOUR_RPY stLrnRpy;
    stLrnCmd.bAutoThreshold = true;
    stLrnCmd.matInputImg = cv::imread ( "./data/RectContourTmpl.png", cv::IMREAD_GRAYSCALE );
    stLrnCmd.rectROI = cv::Rect(0, 0, 471, 372);
    PR_LrnContour(&stLrnCmd, &stLrnRpy);
    PrintLrnReply(stLrnRpy);

    PR_INSP_CONTOUR_CMD stInspCmd;
    PR_INSP_CONTOUR_RPY stInspRpy;
    stInspCmd.matInputImg = cv::imread ( "./data/RectContourTmpl.png", cv::IMREAD_GRAYSCALE );
    stInspCmd.rectROI = cv::Rect(0, 0, 471, 372);
    stInspCmd.nRecordId = stLrnRpy.nRecordId;
    PR_InspContour(&stInspCmd, &stInspRpy);
    PrintInspReply(stInspRpy);

    stInspCmd.matInputImg = cv::imread ( "./data/RectContourInsp.png", cv::IMREAD_GRAYSCALE );
    stInspCmd.rectROI = cv::Rect(0, 0, 471, 372);
    stInspCmd.nRecordId = stLrnRpy.nRecordId;
    PR_InspContour(&stInspCmd, &stInspRpy);
    PrintInspReply(stInspRpy);

    PR_FreeRecord ( stLrnRpy.nRecordId );
}

void TestInspContour() {
    TestInspContour_1();
    TestInspContour_2();
}

}
}