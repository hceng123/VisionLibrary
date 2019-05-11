#include <iostream>
#include <numeric>
#include <iomanip>

#include "CalcUtils.hpp"
#include "Fitting.hpp"
#include "Unwrap.h"
#include "Auxiliary.hpp"
#include "CudaAlgorithm.h"
#include "cuda_runtime.h"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaarithm.hpp"

namespace AOI
{
namespace Vision
{

template<class T>
static void printfVector(const std::vector<T> &vecInput)
{
    for (const auto value : vecInput)
    {
        printf ( "%.2f ", value );
    }
    printf ( "\n" );
}

template<class T>
static void printfVectorOfVector(const std::vector<std::vector<T>> &vecVecInput)
{
    for (const auto &vecInput : vecVecInput)
    {
        for (const auto value : vecInput)
        {
            printf("%.2f ", value);
        }
        printf("\n");
    }
}

template<class T>
static String formatRect(const cv::Rect_<T> &rect)
{
    std::stringstream ss;
    ss << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height;
    return ss.str();
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
        cv::Mat matInput(vecFloat), matResult;
        matInput = matInput.reshape(1, 4);
        std::cout << "INPUT: " << std::endl;
        printfMat<float>(matInput);

        std::cout << "CALCUTILS CUMSUM ON DIMENSION 0 RESULT: " << std::endl;
        matResult = CalcUtils::cumsum<float>(matInput, 0);
        printfMat<float>(matResult);

        std::cout << "CALCUTILS CUMSUM ON DIMENSION 1 RESULT: " << std::endl;
        matResult = CalcUtils::cumsum<float>(matInput, 1);
        printfMat<float>(matResult);
    }
}

static void TestCalcUtilsIntervals() {
    float start = 0.f, interval = 0.f, end = 0.f;
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS INTERVALS TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        start = 0.f; interval = 0.1f; end = 1.f;
        auto matResult = CalcUtils::intervals<float>(start, interval, end);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "INPUT: start = " << start << " interval = " << interval << " end = " << end << std::endl;
        std::cout << "INTERVALS RESULT: " << std::endl;
        printfMat<float>(matResult);
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
        printfMat<float>(matResult);
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
        printfMat<float>(matResult);
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
        cv::Mat matInput(vecInput), matResult;
        matInput = matInput.reshape(1, 3);
        std::cout << "INPUT: " << std::endl;
        printfMat<float>(matInput);

        matResult = CalcUtils::diff(matInput, 1, 1);
        std::cout << "Diff recursive time 1, dim 1 result: " << std::endl;
        printfMat<float>(matResult);

        matResult = CalcUtils::diff(matInput, 1, 2);
        std::cout << "Diff recursive time 1, dim 2 result: " << std::endl;
        printfMat<float>(matResult);
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
        cv::Mat matInput(vecInput), matResult;
        matInput = matInput.reshape(1, 4);
        std::cout << "INPUT: " << std::endl;
        printfMat<float>(matInput);

        matResult = CalcUtils::diff(matInput, 1, 1);
        std::cout << "Diff recursive time 1, dim 1 result: " << std::endl;
        printfMat<float>(matResult);

        matResult = CalcUtils::diff(matInput, 1, 2);
        std::cout << "Diff recursive time 1, dim 2 result: " << std::endl;
        printfMat<float>(matResult);

        matResult = CalcUtils::diff(matInput, 2, 1);
        std::cout << "Diff recursive time 2, dim 1 result: " << std::endl;
        printfMat<float>(matResult);

        matResult = CalcUtils::diff(matInput, 2, 2);
        std::cout << "Diff recursive time 2, dim 2 result: " << std::endl;
        printfMat<float>(matResult);
    }
}

static void TestCalcUtilsDiffGpu() {
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS DIFF GPU TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        std::vector<float> vecInput{
            1, 3, 5,
            7, 11, 13,
            17, 19, 23
        };
        cv::Mat matInput(vecInput), matResult;
        matInput = matInput.reshape(1, 3);
        std::cout << "INPUT: " << std::endl;
        printfMat<float>(matInput);

        cv::cuda::GpuMat matGpuInput(matInput), matGpuOutput;

        matGpuOutput = CudaAlgorithm::diff(matGpuInput, 1, 1);
        matGpuOutput.download(matResult);
        std::cout << "Diff recursive time 1, dim 1 result: " << std::endl;
        printfMat<float>(matResult);

        matGpuOutput = CudaAlgorithm::diff(matGpuInput, 1, 2);
        matGpuOutput.download(matResult);
        std::cout << "Diff recursive time 1, dim 2 result: " << std::endl;
        printfMat<float>(matResult);
    }

    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS DIFF TEST #2 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        std::vector<float> vecInput{
            1, 3, 5, -6,
            7, 11, 13, 16,
            17, 19, 23, -1,
            -5, -7, 9, -21,
        };
        cv::Mat matInput(vecInput), matResult;
        matInput = matInput.reshape(1, 4);
        std::cout << "INPUT: " << std::endl;
        printfMat<float>(matInput);

        cv::cuda::GpuMat matGpuInput(matInput), matGpuOutput;

        matGpuOutput = CudaAlgorithm::diff(matGpuInput, 1, 1);
        matGpuOutput.download(matResult);
        std::cout << "Diff recursive time 1, dim 1 result: " << std::endl;
        printfMat<float>(matResult);

        matGpuOutput = CudaAlgorithm::diff(matGpuInput, 1, 2);
        matGpuOutput.download(matResult);
        std::cout << "Diff recursive time 1, dim 2 result: " << std::endl;
        printfMat<float>(matResult);

        matGpuOutput = CudaAlgorithm::diff(matGpuInput, 2, 1);
        matGpuOutput.download(matResult);
        std::cout << "Diff recursive time 2, dim 1 result: " << std::endl;
        printfMat<float>(matResult);

        matGpuOutput = CudaAlgorithm::diff(matGpuInput, 2, 2);
        matGpuOutput.download(matResult);
        std::cout << "Diff recursive time 2, dim 2 result: " << std::endl;
        printfMat<float>(matResult);
    }
}

static void TestCalcUtilsPower() {
    {
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl << "CALCUTILS POWER TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------";
        std::cout << std::endl;
        std::vector<float> vecInput {
            1,  3,  5,
            7,  11, 13,
            17, 19, 23
        };

        std::vector<float> vecPow {
            1, 2, 1,
            2, 1, 1,
            1, 1, 2
        };

        cv::Mat matInput(vecInput), matPow(vecPow), matResult;
        matInput = matInput.reshape(1, 3);
        std::cout << "INPUT: " << std::endl;
        printfMat<float>(matInput);

        std::cout << "POWER: " << std::endl;
        matPow = matPow.reshape(1, 3);
        printfMat<float>(matPow);

        matResult = CalcUtils::power<float>(matInput, matPow);
        std::cout << "Power result: " << std::endl;
        printfMat<float>(matResult);
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

static void TestCalcUtilsInterp1() {
{
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "CALCUTILS INTERP1 REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    VectorOfDouble vecV( {0,  1.41,  2,  1.41,  0,  -1.41,  -2,  -1.41, 0});
    std::cout << "Input: " << std::endl;
    printfVector<double>(vecV);
    VectorOfDouble vecX(vecV.size());
    std::iota ( vecX.begin(), vecX.end(), 1 );
    VectorOfDouble vecXq;
    for ( double xq = 1.5; xq <= 8.5; xq += 1 ) {
        vecXq.push_back ( xq );
    }
    auto vecVq = CalcUtils::interp1 ( vecX, vecV, vecXq );
    std::cout << "Linear interp1 result: " << std::endl;
    printfVector ( vecVq );

    vecVq = CalcUtils::interp1 ( vecX, vecV, vecXq, true );
    std::cout << "Spine interp1 result: " << std::endl;
    printfVector ( vecVq );
}
}

static void TestCalcUtilsRepeat() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "CALCUTILS REPEAT REGRESSION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    std::vector<float> vecRow ( 5 );
    std::iota ( vecRow.begin(), vecRow.end(), 1.f );
    cv::Mat matRow(vecRow);
    cv::Mat matResult = CalcUtils::repeatInX<float> ( matRow, 5 );
    std::cout << "Repeat in X result: " << std::endl;
    printfMat<float> ( matResult );
}

static void TestCalcUtilsScaleRect() {
    {
        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl << "CALCUTILS SCALE RECT TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl;
        cv::Rect2f rect(-5, -5, 10, 10);
        cv::Rect2f rectBig = CalcUtils::scaleRect(rect, 2.f);
        cv::Rect2f rectSmall = CalcUtils::scaleRect(rect, 0.5f);
        std::cout << "Input rect " << formatRect(rect) << std::endl;
        std::cout << "Scale 2 result " << formatRect(rectBig) << std::endl;
        std::cout << "Scale 0.5 result " << formatRect(rectSmall) << std::endl;
    }

    {
        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl << "CALCUTILS SCALE RECT TEST #2 STARTING";
        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl;

        cv::Rect rect(90, 90, 20, 20);
        cv::Rect rectBig = CalcUtils::scaleRect(rect, 2.f);
        cv::Rect rectSmall = CalcUtils::scaleRect(rect, 0.5f);
        std::cout << "Input rect " << formatRect(rect) << std::endl;
        std::cout << "Scale 2 result " << formatRect(rectBig) << std::endl;
        std::cout << "Scale 0.5 result " << formatRect(rectSmall) << std::endl;
    }
}

static void TestPhaseCorrection() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "UNWRAP PHASE CORRECTION TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    VectorOfVectorOfFloat vecVecPhase = {
        {0.1f, 5.0f, 5.3f, 2.f, 0.0f},
        {-0.1f, 1.1f, 1.3f, 2.f, 0.5f},
        {1.2f, 1.1f, 1.3f, 2.f, 8.5f},
        {1.3f, 0.1f, 0.1f, 2.f, -5.5f},
        {0.2f, 1.1f, 1.3f, 2.f, 0.5f},
    };
    cv::Mat matPhase = CalcUtils::vectorToMat<float>(vecVecPhase);
    std::cout << "Phase Correction input: " << std::endl;
    printfMat<float>(matPhase);
    Unwrap::phaseCorrection(matPhase, 5, 5);
    std::cout << "Phase Correction result: " << std::endl;
    printfMat<float>(matPhase);
}

static void TestFitCircle() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "FIT CIRCLE TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    VectorOfPoint vecPoints;
    vecPoints.emplace_back(636, 1308);
    vecPoints.emplace_back(629, 1307);
    vecPoints.emplace_back(621, 1307);
    vecPoints.emplace_back(615, 1304);
    vecPoints.emplace_back(608, 1302);
    vecPoints.emplace_back(601, 1300);
    vecPoints.emplace_back(594, 1298);
    vecPoints.emplace_back(587, 1294);
    vecPoints.emplace_back(581, 1290);
    vecPoints.emplace_back(576, 1286);
    vecPoints.emplace_back(570, 1281);
    vecPoints.emplace_back(565, 1275);
    vecPoints.emplace_back(561, 1270);
    vecPoints.emplace_back(557, 1264);
    vecPoints.emplace_back(553, 1257);
    vecPoints.emplace_back(550, 1251);
    vecPoints.emplace_back(549, 1243);
    vecPoints.emplace_back(547, 1236);
    vecPoints.emplace_back(545, 1229);
    vecPoints.emplace_back(544, 1222);

    auto fitResult = Fitting::fitCircle(vecPoints);
    std::cout << "Fit circle center " << fitResult.center << ", Radius " << fitResult.size.width / 2.f << std::endl;
}

static void TestBilinearInterpolation() {
    {
        VectorOfPoint2f vecPoints{cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 100.f), cv::Point2f(100.f, 100.f), cv::Point2f(100.f, 0.f)};
        VectorOfFloat vecValues{10.f, 20.f, 30.f, 40.f};

        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl << "BILINEAR INTERPOLATION TEST #1 STARTING";
        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl;

        cv::Point2f ptToGet = cv::Point2f (100.f, 0.f);
        float fResult = 0.f;
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;

        ptToGet = cv::Point2f (50.f, 50.f);
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;

        ptToGet = cv::Point2f (10.f, 15.f);
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;

        ptToGet = cv::Point2f (99.f, 99.f);
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;
    }

     {
        VectorOfPoint2f vecPoints{cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(100.f, 0.f), cv::Point2f(100.f, 0.f)};
        VectorOfFloat vecValues{10.f, 10.f, 20.f, 20.f};

        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl << "BILINEAR INTERPOLATION TEST #2 STARTING";
        std::cout << std::endl << "------------------------------------------------";
        std::cout << std::endl;

        cv::Point2f ptToGet = cv::Point2f (100.f, 0.f);
        float fResult = 0.f;
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;

        ptToGet = cv::Point2f (20.f, 0.f);
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;

        ptToGet = cv::Point2f (50.f, 0.f);
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;

        ptToGet = cv::Point2f (70.f, 0.f);
        fResult = CalcUtils::bilinearInterpolate(vecPoints, vecValues, ptToGet);
        std::cout << "Point " << ptToGet << " result " << fResult << std::endl;
    }
}

static void TestGpuMatFloor() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "CUDA GPU FLOOR TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    cv::Mat matDataCpu(2048, 2040, CV_32FC1);
    matDataCpu = 1.7f;

    cv::cuda::GpuMat matDataGpu;
    matDataGpu.upload(matDataCpu);

    CudaAlgorithm::floor(matDataGpu);
    matDataGpu.download(matDataCpu);

    cv::Mat matDiff = matDataCpu != 1.f;
    int notCorrectCount = cv::countNonZero(matDiff);
    if (notCorrectCount == 0) {
        std::cout << "Test passed" << std::endl;
    }else {
        std::cout << "Test failed, failed count " << notCorrectCount << std::endl;
    }
}

static void TestIntervalAverage() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "CUDA GPU AVERAGE INTERVAL TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    cv::Mat matDataCpu(2048, 2040, CV_32FC1);
    matDataCpu = 1.7f;

    cv::cuda::GpuMat matDataGpu;
    matDataGpu.upload(matDataCpu);

    float* d_result;
    cudaMalloc(reinterpret_cast<void **>(&d_result), sizeof(float));
    float result = CudaAlgorithm::intervalAverage(matDataGpu, 20, d_result);
    cudaFree(d_result);

    std::cout << "Average result " << result << std::endl;
}

static void TestRangeIntervalAverage() {
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl << "CUDA GPU AVERAGE RANGE INTERVAL TEST #1 STARTING";
    std::cout << std::endl << "------------------------------------------------";
    std::cout << std::endl;

    cv::Mat matDataCpu(2048, 2040, CV_32FC1);
    matDataCpu = 1.5f;
    cv::Mat matHalf(matDataCpu, cv::Range::all(), cv::Range(1020, 2040));
    matHalf = 2.5f;

    cv::cuda::GpuMat matDataGpu;
    matDataGpu.upload(matDataCpu);

    float* d_result;
    cudaMalloc(reinterpret_cast<void **>(&d_result), sizeof(float));
    float result = CudaAlgorithm::intervalRangeAverage(matDataGpu, 10, 0.4f, 0.6f, d_result);
    cudaFree(d_result);

    std::cout << "Average result " << result << std::endl;
}

static void TestCudaPhaseToHeight3D() {
    std::cout << std::endl << "----------------------------------------------------------";
    std::cout << std::endl << "CUDA PHASE TO HEIGHT 3D INTERVAL TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------------------";
    std::cout << std::endl;

    const int ROWS = 2040;
    const int COLS = 2048;
    const int TOTAL = ROWS * COLS;
    const int ss = 4;

    cv::cuda::GpuMat zInLine(1, TOTAL, CV_32FC1);
    zInLine.setTo(0.5f);
    auto matPolyParass = CalcUtils::paraFromPolyNomial(ss);
    std::cout << "matPolyParass " << ss << std::endl;
    printfMat<float>(matPolyParass, 2);

    cv::cuda::GpuMat matPolyParassGpu(matPolyParass);

    //cv::cuda::GpuMat xxt(TOTAL, ss, CV_32FC1);
    //xxt.setTo(3);
    const int MEM_SIZE = TOTAL * ss * sizeof(float);
    float* h_xxt = (float *)malloc(MEM_SIZE);
    for (int i = 0; i < TOTAL * ss; ++i)
        h_xxt[i] = 3.f;

    float* d_xxt;
    cudaMalloc(reinterpret_cast<void **>(&d_xxt), MEM_SIZE);
    cudaMemcpy(d_xxt, h_xxt, MEM_SIZE, cudaMemcpyHostToDevice);
    
    float* d_P3;
    cudaMalloc(reinterpret_cast<void **>(&d_P3), MEM_SIZE);

    std::cout << "xxt step " << ss << std::endl;

    CudaAlgorithm::phaseToHeight3D(
        zInLine,
        matPolyParassGpu,
        d_xxt,
        ss,
        TOTAL, ss,
        d_P3);

    cudaDeviceSynchronize();

    float* h_data = (float *)malloc(MEM_SIZE);
    cudaMemcpy(h_data, d_P3, MEM_SIZE, cudaMemcpyDeviceToHost);

    std::cout << "Check result data " << std::endl;
    for (int i = 0; i < TOTAL; i += 10000) {
        float* pRow = h_data + i * ss;
        for (int col = 0; col < ss; ++ col) {
            std::cout << pRow[col] << " ";
        }
        std::cout << std::endl;
    }

    free(h_data);
    free(h_xxt);
    cudaFree(d_P3);
    cudaFree(d_xxt);
}

static void TestCalcSumAndConvertMatrix() {
    std::cout << std::endl << "----------------------------------------------------------";
    std::cout << std::endl << "CUDA CALC SUM AND CONVERT MATRIX INTERVAL TEST #1 STARTING";
    std::cout << std::endl << "----------------------------------------------------------";
    std::cout << std::endl;
    const int ROWS = 2040;
    const int COLS = 2048;
    const int TOTAL = ROWS * COLS;
    const int ss = 4;
    const int MEM_SIZE = TOTAL * ss * sizeof(float);
    float* h_data = (float *)malloc(MEM_SIZE);
    for (int row = 0; row < TOTAL; ++ row) {
        float* pResultRow = h_data + row * ss;
        for (int col = 0; col < ss; ++ col) {
            pResultRow[col] = 50.f;
        }
    }

    float* d_data;
    cudaMalloc(reinterpret_cast<void **>(&d_data), MEM_SIZE);
    cudaMemcpy(d_data, h_data, MEM_SIZE, cudaMemcpyHostToDevice);

    cv::cuda::GpuMat matResultGpu(ROWS, COLS, CV_32FC1);
    CudaAlgorithm::calcSumAndConvertMatrix(d_data, ss, matResultGpu);
    cudaDeviceSynchronize();

    cv::Mat matResult;
    matResultGpu.download(matResult);
    std::cout << "Mean of result " << cv::mean(matResult)[0] << std::endl;
    matResult.convertTo(matResult, CV_8UC1);

    free(h_data);
    cudaFree(d_data);
}

void InternalTest() {
    TestBilinearInterpolation();
    TestCalcUtilsCumSum();
    TestCalcUtilsIntervals();
    TestCalcUtilsMeshGrid();
    TestCalcUtilsDiff();
    TestCalcUtilsDiffGpu();
    TestCalcUtilsPower();
    TestCountOfNan();
    TestFindLargestKItems();
    TestCalcUtilsFloor();
    TestCalcUtilsInterp1();
    TestCalcUtilsRepeat();
    TestCalcUtilsScaleRect();
    TestPhaseCorrection();
    TestFitCircle();
    TestGpuMatFloor();
    TestIntervalAverage();
    TestRangeIntervalAverage();
    TestCudaPhaseToHeight3D();
    TestCalcSumAndConvertMatrix();
}

}
}