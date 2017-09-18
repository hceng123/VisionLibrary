#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>
#include <iomanip>

namespace AOI
{
namespace Vision
{

static void PrintMatchTmplRpy(const PR_MATCH_TEMPLATE_RPY &stRpy) {
    std::cout << "Match template status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Match template result " << stRpy.ptObjPos.x << ", " << stRpy.ptObjPos.y << std::endl;
    std::cout << "Match template angle " << stRpy.fRotation << std::endl;
}

void TestTmplMatch_1()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/Template.png", cv::IMREAD_GRAYSCALE);
    stLrnCmd.rectROI = cv::Rect (80, 80, 100, 100 );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/FindAngle_1.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.bSubPixelRefine = true;
    stCmd.enMotion = PR_OBJECT_MOTION::EUCLIDEAN;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch_2()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/CapsLock_1.png");
    stLrnCmd.rectROI = cv::Rect (0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/CapsLock_2.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::HIERARCHICAL_EDGE;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch_3()
{
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "MATCH TEMPLATE REGRESSION TEST #3 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    PR_LRN_TEMPLATE_CMD stLrnCmd;
    PR_LRN_TEMPLATE_RPY stLrnRpy;
    stLrnCmd.matInputImg = cv::imread("./data/CapsLock_1.png");
    stLrnCmd.rectROI = cv::Rect (0, 0, stLrnCmd.matInputImg.cols, stLrnCmd.matInputImg.rows );
    stLrnCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF;
    PR_LrnTmpl ( &stLrnCmd, &stLrnRpy );
    if ( stLrnRpy.enStatus != VisionStatus::OK ) {
        std::cout << "Failed to learn template." << std::endl;
        return;
    }

    PR_MATCH_TEMPLATE_CMD stCmd;
    PR_MATCH_TEMPLATE_RPY stRpy;
    stCmd.matInputImg = cv::imread("./data/CapsLock_2.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 0, stCmd.matInputImg.cols, stCmd.matInputImg.rows );
    stCmd.enAlgorithm = PR_MATCH_TMPL_ALGORITHM::SQUARE_DIFF;
    stCmd.nRecordId = stLrnRpy.nRecordId;
    stCmd.enMotion = PR_OBJECT_MOTION::TRANSLATION;
    
    PR_MatchTmpl(&stCmd, &stRpy);
    PrintMatchTmplRpy ( stRpy );
}

void TestTmplMatch() {
    TestTmplMatch_1();
    TestTmplMatch_2();
    TestTmplMatch_3();
}

static void PrintFiducialMarkResult(const PR_SRCH_FIDUCIAL_MARK_RPY &stRpy) {
    std::cout << "Search fiducial status " << ToInt32(stRpy.enStatus) << std::endl;
    std::cout << "Search fiducial result " << stRpy.ptPos.x << ", " << stRpy.ptPos.y << std::endl;
}

void TestSrchFiducialMark()
{
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "SEARCH FIDUCIAL MARK REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
    PR_SRCH_FIDUCIAL_MARK_RPY stRpy;
    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::SQUARE;
    stCmd.fSize = 54;
    stCmd.fMargin = 8;
    stCmd.matInputImg = cv::imread("./data/F6-313-1-Gray.bmp", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(1459, 155, 500, 500 );
    
    PR_SrchFiducialMark(&stCmd, &stRpy);
    PrintFiducialMarkResult ( stRpy );

    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "SEARCH FIDUCIAL MARK REGRESSION TEST #2 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    stCmd.enType = PR_FIDUCIAL_MARK_TYPE::CIRCLE;
    stCmd.fSize = 64;
    stCmd.fMargin = 8;
    stCmd.matInputImg = cv::imread("./data/CircleFiducialMark.png", cv::IMREAD_GRAYSCALE);
    stCmd.rectSrchWindow = cv::Rect(0, 40, 250, 250 );

    PR_SrchFiducialMark(&stCmd, &stRpy);
    PrintFiducialMarkResult ( stRpy );    
}

}
}