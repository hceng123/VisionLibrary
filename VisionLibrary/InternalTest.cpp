#include "Fitting.h"
#include "CalcUtils.h"
#include <iostream>

namespace AOI
{
namespace Vision
{

template<class T>
static void printfMat ( const cv::Mat &mat)
{
    for ( short row = 0; row < mat.rows; ++ row )
    {
        for ( short col = 0; col < mat.cols; ++ col )
        {
            printf ("%.2f ", mat.at<T>(row, col) );
        }
        printf("\n");
    }
}

template<class T>
static void printfVectorOfVector(const std::vector<std::vector<T>> &vevVecInput)
{
	for (const auto &vecInput : vevVecInput )
	{
		for (const auto value : vecInput )
		{
			printf("%.2f ", value);
		}
		printf("\n");
	}
}

static void TestCalcUtilsCumSum() {    
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CUMSUM REGRESSION TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        std::vector<float> vecFloat {
            0.0f, 0.8f, 0.2f, 0.3f,
            0.3f, 0.5f, 0.3f, 0.4f,
            0.9f, 0.8f, 0.6f, 0.5f,
            0.8f, 0.8f, 0.7f, 0.6f
        };
        cv::Mat matInput ( vecFloat ), matResult;
        matInput = matInput.reshape ( 1, 4 );
        std::cout << "INPUT: " << std::endl;
        printfMat<float> ( matInput );

        std::cout << "CALCUTILS CUMSUM ON DIMENSION 0 RESULT: " << std::endl;
        matResult = CalcUtils::cumsum<float> ( matInput, 0 );
        printfMat<float> ( matResult );

        std::cout << "CALCUTILS CUMSUM ON DIMENSION 1 RESULT: " << std::endl;
        matResult = CalcUtils::cumsum<float> ( matInput, 1 );
        printfMat<float> ( matResult );
    }
}

static void TestCalcUtilsInternals() {
    float start = 0.f, interval = 0.f, end = 0.f;
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS INTERVALS TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        start = 0.f; interval = 0.1f; end = 1.f;
        auto matResult = CalcUtils::intervals<float>(start, interval, end);
        std::cout << "INPUT: start = " << start << "interval = " << interval << " end = " << end << std::endl;
        std::cout << "INTERVALS RESULT: " << std::endl;
        printfMat<float> ( matResult );
    }

    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS INTERVALS TEST #2 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        start = 0.f; interval = -0.1f; end = -1.f;
        auto matResult = CalcUtils::intervals<float>(start, interval, end);
        std::cout << "INPUT: start = " << start << " interval = " << interval << " end = " << end << std::endl;
        std::cout << "INTERVALS RESULT: " << std::endl;
        printfMat<float> ( matResult );
    }

    try
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS INTERVALS TEST #3 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        start = 0.f; interval = -0.1f; end = 1.f;
        auto matResult = CalcUtils::intervals<float>(start, interval, end);
        std::cout << "INPUT: start = " << start << " interval = " << interval << " end = " << end << std::endl;
        std::cout << "INTERVALS RESULT: " << std::endl;
        printfMat<float> ( matResult );
    }catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

static void TestCalcUtilsMeshGrid()
{
    cv::Mat matX, matY;
    float xStart = 0.f, xInterval = 0.f, xEnd = 0.f, yStart = 0.f, yInterval = 0.f, yEnd = 0.f;
    char chArrMsg[1000];
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS MESHGRID TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        xStart = -2.f; xInterval = 1.f; xEnd = 2.f; yStart = -2.f; yInterval = 1.f; yEnd = 2.f;
        CalcUtils::meshgrid<float>(xStart, xInterval, xEnd, yStart, yInterval, yEnd, matX, matY );
        _snprintf(chArrMsg, sizeof(chArrMsg), "xStart = %.2f, xInterval = %.2f, xEnd = %.2f, yStart = %.2f, yInterval = %.2f, yEnd = %.2f.", xStart, xInterval, xEnd, yStart, yInterval, yEnd, matX, matY );
        std::cout << "INPUT: " << chArrMsg << std::endl;
        std::cout << "MESHGRID RESULT: " << std::endl;
        std::cout << "MATRIX X: " << std::endl;
        printfMat<float> ( matX );
        std::cout << "MATRIX Y: " << std::endl;
        printfMat<float> ( matY );
    }

    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS MESHGRID TEST #2 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        xStart = -2.f; xInterval = 1.f; xEnd = 2.f; yStart = 2.f; yInterval = 1.f; yEnd = 5.f;
        CalcUtils::meshgrid<float>(xStart, xInterval, xEnd, yStart, yInterval, yEnd, matX, matY );
        _snprintf(chArrMsg, sizeof(chArrMsg), "xStart = %.2f, xInterval = %.2f, xEnd = %.2f, yStart = %.2f, yInterval = %.2f, yEnd = %.2f.", xStart, xInterval, xEnd, yStart, yInterval, yEnd, matX, matY );
        std::cout << "INPUT: " << chArrMsg << std::endl;
        std::cout << "MESHGRID RESULT: " << std::endl;
        std::cout << "MATRIX X: " << std::endl;
        printfMat<float> ( matX );
        std::cout << "MATRIX Y: " << std::endl;
        printfMat<float> ( matY );
    }

    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS MESHGRID TEST #3 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        xStart = 2.f; xInterval = -1.f; xEnd = -2.f; yStart = 5.f; yInterval = -1.f; yEnd = 2.f;
        CalcUtils::meshgrid<float>(xStart, xInterval, xEnd, yStart, yInterval, yEnd, matX, matY );
        _snprintf(chArrMsg, sizeof(chArrMsg), "xStart = %.2f, xInterval = %.2f, xEnd = %.2f, yStart = %.2f, yInterval = %.2f, yEnd = %.2f.", xStart, xInterval, xEnd, yStart, yInterval, yEnd, matX, matY );
        std::cout << "INPUT: " << chArrMsg << std::endl;
        std::cout << "MESHGRID RESULT: " << std::endl;
        std::cout << "MATRIX X: " << std::endl;
        printfMat<float> ( matX );
        std::cout << "MATRIX Y: " << std::endl;
        printfMat<float> ( matY );
    }
}

static void TestCalcUtilsDiff() {
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS DIFF TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        std::vector<float> vecInput {
            1,  3,  5,
            7,  11, 13,
            17, 19, 23
        };
        cv::Mat matInput ( vecInput ), matResult;
        matInput = matInput.reshape ( 1, 3 );
        std::cout << "INPUT: " << std::endl;
        printfMat<float> ( matInput );

        matResult = CalcUtils::diff ( matInput, 1, 1 );
        std::cout << "Diff recursive time 1, dim 1 result: " << std::endl;
        printfMat<float> ( matResult );

        matResult = CalcUtils::diff ( matInput, 1, 2 );
        std::cout << "Diff recursive time 1, dim 2 result: " << std::endl;
        printfMat<float> ( matResult );
    }

    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS DIFF TEST #2 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        std::vector<float> vecInput {
            1,  3,  5,  -6,
            7,  11, 13, 16,
            17, 19, 23, -1,
            -5, -7, 9,  -21,
        };
        cv::Mat matInput ( vecInput ), matResult;
        matInput = matInput.reshape ( 1, 4 );
        std::cout << "INPUT: " << std::endl;
        printfMat<float> ( matInput );

        matResult = CalcUtils::diff ( matInput, 1, 1 );
        std::cout << "Diff recursive time 1, dim 1 result: " << std::endl;
        printfMat<float> ( matResult );

        matResult = CalcUtils::diff ( matInput, 1, 2 );
        std::cout << "Diff recursive time 1, dim 2 result: " << std::endl;
        printfMat<float> ( matResult );

        matResult = CalcUtils::diff ( matInput, 2, 1 );
        std::cout << "Diff recursive time 2, dim 1 result: " << std::endl;
        printfMat<float> ( matResult );

        matResult = CalcUtils::diff ( matInput, 2, 2 );
        std::cout << "Diff recursive time 2, dim 2 result: " << std::endl;
        printfMat<float> ( matResult );
    }
}

static void TestCountOfNan() {
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl << "COUNT OF NAN TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------";
    std::cout << std::endl;

    cv::Mat matNan (3, 3, CV_32FC1, NAN );
    matNan.at<float>(0) = 0;
    std::cout << "Count of Nan: " << CalcUtils::countOfNan(matNan) << std::endl;
}

static void TestFindLargestKItems() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "FIND LARGEST K ITEMS REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;
    std::vector<float> vecFloat{
        NAN, 0.8f, 0.2f, 0.3f,
        0.3f, 0.5f, 0.3f, 0.4f,
        0.9f, -0.8f, 0.6f, 0.5f,
        1.8f, 1.9f, 0.7f, NAN
    };
    cv::Mat matInput ( vecFloat ), matResult;
    matInput = matInput.reshape ( 1, 4 );
    std::cout << "INPUT: " << std::endl;
    printfMat<float> ( matInput );

    std::vector<size_t> vecIndex;
    auto vecValues = CalcUtils::FindLargestKItems<float> ( matInput, 5, vecIndex );
    std::cout << "Largest Item Values: " << std::endl;
    for ( const auto value : vecValues )
        std::cout << value << " ";
    std::cout << std::endl;
    std::cout << "Largest Item Index: " << std::endl;
    for ( const auto value : vecIndex )
        std::cout << value << " ";
    std::cout << std::endl;
}

static void TestCalcUtilsFloor() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "CALCUTILS FLOOR REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;
    std::vector<float> vecFloat {
        NAN, 0.8f, 0.2f, 0.3f,
        0.3f, 0.5f, 0.3f, 0.4f,
        0.9f, -0.8f, 0.6f, 0.5f,
        1.8f, 1.9f, 0.7f, NAN
    };
    cv::Mat matInput ( vecFloat );
    matInput = matInput.reshape ( 1, 4 );
    std::cout << "INPUT: " << std::endl;
    printfMat<float> ( matInput );
    cv::Mat matResult = CalcUtils::floor<float>( matInput );
    std::cout << "CalcUtils::floor result: " << std::endl;
    printfMat<float> ( matResult );
}

void InternalTest() {
    TestCalcUtilsCumSum();
    TestCalcUtilsInternals();
    TestCalcUtilsMeshGrid();
    TestCalcUtilsDiff();
    TestCountOfNan();
    TestFindLargestKItems();
    TestCalcUtilsFloor();
}

}
}