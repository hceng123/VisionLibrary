#include "stdafx.h"
#include "../VisionLibrary/VisionAPI.h"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include "TestSub.h"

using namespace AOI::Vision;

void TestCalibCamera()
{
    std::string strImgPath = "./data/ChessBoard.png";
    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;
    stCmd.matInputImg = cv::imread(strImgPath, cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2.;
    stCmd.szBoardPattern = cv::Size(25, 23); //cv::Size( 5, 5 );

    PR_CalibrateCamera ( &stCmd, &stRpy );
    if ( VisionStatus::OK != stRpy.enStatus ) {
        std::cout << "Failed to calibrate camera!" << std::endl;
        return;
    }

    bool ok = cv::checkRange(stRpy.matIntrinsicMatrix) && cv::checkRange(stRpy.matIntrinsicMatrix);
    std::cout << "OK = " << ok << ", Camera Instrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matIntrinsicMatrix);

    std::cout << "Camera Extrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matExtrinsicMatrix);

    std::cout << "Initial Camera Instrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matInitialIntrinsicMatrix);

    std::cout << "Initial Camera Extrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matInitialExtrinsicMatrix);

    std::cout << "Distortion Coefficient: " << std::endl;
    printfMat<double>(stRpy.matDistCoeffs);

    //PR_CALC_UNDISTORT_RECTIFY_MAP_CMD stCalcUndistortRectifyMapCmd;
    //PR_CALC_UNDISTORT_RECTIFY_MAP_RPY stCalcUndistortRectifyMapRpy;
    //stCalcUndistortRectifyMapCmd.matIntrinsicMatrix = stRpy.matIntrinsicMatrix;
    //stCalcUndistortRectifyMapCmd.matDistCoeffs      = stRpy.matDistCoeffs;
    //stCalcUndistortRectifyMapCmd.szImage            = stCmd.matInputImg.size();
    //PR_CalcUndistortRectifyMap(&stCalcUndistortRectifyMapCmd, &stCalcUndistortRectifyMapRpy);
    //if ( VisionStatus::OK != stCalcUndistortRectifyMapRpy.enStatus ) {
    //    std::cout << "Failed to calculate undistort rectify map!" << std::endl;
    //    return;
    //}

    PR_RESTORE_IMG_CMD stRetoreCmd;
    PR_RESTORE_IMG_RPY stRetoreRpy;
    stRetoreCmd.matInputImg = cv::imread(strImgPath, cv::IMREAD_GRAYSCALE );
    stRetoreCmd.vecMatRestoreImage = stRpy.vecMatRestoreImage; //stCalcUndistortRectifyMapRpy.vecMatRestoreImage;
    PR_RestoreImage ( &stRetoreCmd, &stRetoreRpy );
    if ( VisionStatus::OK != stRetoreRpy.enStatus) {
        std::cout << "Failed to restore image!" << std::endl;
        return;
    }

    cv::Mat matInputFloat;
    stRetoreCmd.matInputImg.convertTo ( matInputFloat, CV_32FC1 );

    //cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - stRetoreRpy.matResultImg );
    cv::imwrite("./data/chessboardRestored.png", stRetoreRpy.matResultImg);

    cv::Mat matReadRestored = cv::imread("./data/chessboardRestored.png", cv::IMREAD_GRAYSCALE );
    cv::Mat matReadAgainFloat;
    matReadRestored.convertTo ( matReadAgainFloat, CV_32FC1 );

    cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - matReadAgainFloat );
    cv::imwrite("./data/chessboard_diff.png", matDiff);
    PR_DumpTimeLog("./data/TimeLog.log");
}

void TestCalibCamera_1()
{
    std::string strImgPath = "./data/CLNChessboardWith2DCode.png";
    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;
    stCmd.matInputImg = cv::imread(strImgPath, cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2.;
    stCmd.fMinTmplMatchScore = 80;
    stCmd.szBoardPattern = cv::Size(16, 16); //cv::Size( 5, 5 );

    PR_CalibrateCamera ( &stCmd, &stRpy );
    if ( VisionStatus::OK != stRpy.enStatus ) {
        std::cout << "Failed to calibrate camera!" << std::endl;
        return;
    }

    bool ok = cv::checkRange(stRpy.matIntrinsicMatrix) && cv::checkRange(stRpy.matIntrinsicMatrix);
    std::cout << "OK = " << ok << ", Camera Instrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matIntrinsicMatrix);

    std::cout << "Camera Extrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matExtrinsicMatrix);

    std::cout << "Initial Camera Instrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matInitialIntrinsicMatrix);

    std::cout << "Initial Camera Extrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matInitialExtrinsicMatrix);

    std::cout << "Distortion Coefficient: " << std::endl;
    printfMat<double>(stRpy.matDistCoeffs);

    //PR_CALC_UNDISTORT_RECTIFY_MAP_CMD stCalcUndistortRectifyMapCmd;
    //PR_CALC_UNDISTORT_RECTIFY_MAP_RPY stCalcUndistortRectifyMapRpy;
    //stCalcUndistortRectifyMapCmd.matIntrinsicMatrix = stRpy.matIntrinsicMatrix;
    //stCalcUndistortRectifyMapCmd.matDistCoeffs      = stRpy.matDistCoeffs;
    //stCalcUndistortRectifyMapCmd.szImage            = stCmd.matInputImg.size();
    //PR_CalcUndistortRectifyMap(&stCalcUndistortRectifyMapCmd, &stCalcUndistortRectifyMapRpy);
    //if ( VisionStatus::OK != stCalcUndistortRectifyMapRpy.enStatus ) {
    //    std::cout << "Failed to calculate undistort rectify map!" << std::endl;
    //    return;
    //}

    cv::Mat matPoint( stRpy.vecObjectPoints );
    //cv::transpose ( matPoint, matPoint );
    //int nChannels = matPoint.channels();
    matPoint = matPoint.reshape(1, matPoint.rows);
    cv::transpose ( matPoint, matPoint );
    cv::Mat mat2 = cv::Mat::ones(1, matPoint.cols, matPoint.type() );
    matPoint.push_back ( mat2 );
    matPoint.convertTo ( matPoint, CV_64FC1 );
    auto vevVecPoints = matToVector<double>(matPoint);

    {
        std::ofstream ofStrm;
        ofStrm.open("./data/UVCoordinate.txt");
        for (size_t index = 0; index < stRpy.vecImagePoints.size(); ++index) {
            cv::Point2f ptImage = stRpy.vecImagePoints[index];
            ofStrm << ptImage.x << ", " << ptImage.y << std::endl;
        }
        ofStrm.close();    
    }
    {
        std::ofstream ofStrm;
        ofStrm.open("./data/XYZCoordinate.txt");
        for (size_t index = 0; index < stRpy.vecObjectPoints.size(); ++index) {
            cv::Point3f ptObject = stRpy.vecObjectPoints[index];
            ofStrm << ptObject.x << ", " << ptObject.y << ", " << ptObject.z << std::endl;
        }
        ofStrm.close();    
    }
    {
        cv::Mat matxyz = stRpy.matExtrinsicMatrix * matPoint;
        double dZ = matxyz.at<double>(2, 0);
        cv::Mat matxyz1 = matxyz / dZ;
        cv::Mat matX1(matxyz1, cv::Rect(0, 0, matxyz1.cols, 1));
        cv::Mat matY1(matxyz1, cv::Rect(0, 1, matxyz1.cols, 1));
        cv::Mat matxyz1_Square = matxyz1.mul(matxyz1);
        cv::Mat matxyz1_Square_ROI(matxyz1_Square, cv::Rect(0, 0, matxyz1_Square.cols, 2));
        cv::Mat matR_2;
        cv::reduce(matxyz1_Square_ROI, matR_2, 0, cv::ReduceTypes::REDUCE_SUM);
        cv::Mat matR_4 = matR_2.mul(matR_2);
        cv::Mat matR_6 = matR_4.mul(matR_2);

        double k1 = stRpy.matDistCoeffs.at<double>(0, 0);
        double k2 = stRpy.matDistCoeffs.at<double>(0, 1);
        double p1 = stRpy.matDistCoeffs.at<double>(0, 2);
        double p2 = stRpy.matDistCoeffs.at<double>(0, 3);
        double k3 = stRpy.matDistCoeffs.at<double>(0, 4);

        cv::Mat matFactor = cv::Mat::ones(1, matR_2.cols, matR_2.type()) + k1 * matR_2 + k2 * matR_4 + k3 * matR_6;

        cv::Mat matX2 = matX1.mul(matFactor) + 2. * p1 * matX1.mul(matY1) + p2 * (matR_2 + 2. * matX1.mul(matX1));
        cv::Mat matY2 = matY1.mul(matFactor) + p1 * (matR_2 + 2. * matY1.mul(matY1)) + 2. * p2 * matX1.mul(matY1);

        cv::Mat matXYZ_2 = matX2.clone();
        matXYZ_2.push_back(matY2);
        cv::Mat matZ_2 = cv::Mat::ones(1, matXYZ_2.cols, matXYZ_2.type());
        matXYZ_2.push_back(matZ_2);
        cv::Mat matUV = stRpy.matIntrinsicMatrix * matXYZ_2;
        auto vecVecProjectPoints = matToVector<double>(matUV);

        std::ofstream ofStrm;
        ofStrm.open("./data/CoordinateDiffFinalMatrix.txt");
        for (size_t index = 0; index < stRpy.vecImagePoints.size(); ++index) {
            cv::Point2f ptImage = stRpy.vecImagePoints[index];
            cv::Point2f ptProject = cv::Point2f ( vecVecProjectPoints[0][index], vecVecProjectPoints[1][index]);
            ofStrm << ptImage.x - ptProject.x << ", " << ptImage.y - ptProject.y << std::endl;
        }
        ofStrm.close();
    }

    {
        cv::Mat matProject = stRpy.matInitialIntrinsicMatrix * stRpy.matInitialExtrinsicMatrix * matPoint;
        double dScale = matProject.at<double>(2, 0);
        matProject = matProject / dScale;
        auto vecVecProjectPoints = matToVector<double>(matProject);

        std::ofstream ofStrm;
        ofStrm.open("./data/CoordinateDiffInitialMatrix.txt");
        for (size_t index = 0; index < stRpy.vecImagePoints.size(); ++index) {
            cv::Point2f ptImage = stRpy.vecImagePoints[index];
            cv::Point2f ptProject = cv::Point2f ( vecVecProjectPoints[0][index], vecVecProjectPoints[1][index] );
            ofStrm << ptImage.x - ptProject.x << ", " << ptImage.y - ptProject.y << std::endl;
        }
        ofStrm.close();
    }
    //PR_RESTORE_IMG_CMD stRetoreCmd;
    //PR_RESTORE_IMG_RPY stRetoreRpy;
    //stRetoreCmd.matInputImg = cv::imread(strImgPath, cv::IMREAD_GRAYSCALE );
    //stRetoreCmd.vecMatRestoreImage = stRpy.vecMatRestoreImage; //stCalcUndistortRectifyMapRpy.vecMatRestoreImage;
    //PR_RestoreImage ( &stRetoreCmd, &stRetoreRpy );
    //if ( VisionStatus::OK != stRetoreRpy.enStatus) {
    //    std::cout << "Failed to restore image!" << std::endl;
    //    return;
    //}

    //cv::Mat matInputFloat;
    //stRetoreCmd.matInputImg.convertTo ( matInputFloat, CV_32FC1 );

    ////cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - stRetoreRpy.matResultImg );
    //cv::imwrite("./data/chessboardRestored.png", stRetoreRpy.matResultImg);

    //cv::Mat matReadRestored = cv::imread("./data/chessboardRestored.png", cv::IMREAD_GRAYSCALE );
    //cv::Mat matReadAgainFloat;
    //matReadRestored.convertTo ( matReadAgainFloat, CV_32FC1 );

    //cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - matReadAgainFloat );
    //cv::imwrite("./data/chessboard_diff.png", matDiff);
    PR_DumpTimeLog("./data/TimeLog.log");
}

void TestCalibCamera_2()
{
    std::string strImgPath = "./data/CLN_Chessboard_2.png";
    PR_CALIBRATE_CAMERA_CMD stCmd;
    PR_CALIBRATE_CAMERA_RPY stRpy;
    stCmd.matInputImg = cv::imread(strImgPath, cv::IMREAD_GRAYSCALE );
    stCmd.fPatternDist = 2.;
    stCmd.fMinTmplMatchScore = 85;
    stCmd.szBoardPattern = cv::Size(10, 9); //cv::Size( 5, 5 );

    PR_CalibrateCamera ( &stCmd, &stRpy );
    if ( VisionStatus::OK != stRpy.enStatus ) {
        std::cout << "Failed to calibrate camera!" << std::endl;
        return;
    }

    bool ok = cv::checkRange(stRpy.matIntrinsicMatrix) && cv::checkRange(stRpy.matIntrinsicMatrix);
    std::cout << "OK = " << ok << ", Camera Instrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matIntrinsicMatrix);

    std::cout << "Camera Extrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matExtrinsicMatrix);

    std::cout << "Initial Camera Instrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matInitialIntrinsicMatrix);

    std::cout << "Initial Camera Extrinsic Matrix: " << std::endl;
    printfMat<double>(stRpy.matInitialExtrinsicMatrix);

    std::cout << "Distortion Coefficient: " << std::endl;
    printfMat<double>(stRpy.matDistCoeffs);

    cv::Mat matPoint( stRpy.vecObjectPoints );
    matPoint = matPoint.reshape(1, matPoint.rows);
    cv::transpose ( matPoint, matPoint );
    cv::Mat mat2 = cv::Mat::ones(1, matPoint.cols, matPoint.type() );
    matPoint.push_back ( mat2 );
    matPoint.convertTo ( matPoint, CV_64FC1 );
    auto vevVecPoints = matToVector<double>(matPoint);

    {
        std::ofstream ofStrm;
        ofStrm.open("./data/UVCoordinate.txt");
        for (size_t index = 0; index < stRpy.vecImagePoints.size(); ++index) {
            cv::Point2f ptImage = stRpy.vecImagePoints[index];
            ofStrm << ptImage.x << ", " << ptImage.y << std::endl;
        }
        ofStrm.close();    
    }
    {
        std::ofstream ofStrm;
        ofStrm.open("./data/XYZCoordinate.txt");
        for (size_t index = 0; index < stRpy.vecObjectPoints.size(); ++index) {
            cv::Point3f ptObject = stRpy.vecObjectPoints[index];
            ofStrm << ptObject.x << ", " << ptObject.y << ", " << ptObject.z << std::endl;
        }
        ofStrm.close();    
    }
    {
        cv::Mat matxyz = stRpy.matExtrinsicMatrix * matPoint;
        double dZ = matxyz.at<double>(2, 0);
        cv::Mat matxyz1 = matxyz / dZ;
        cv::Mat matX1(matxyz1, cv::Rect(0, 0, matxyz1.cols, 1));
        cv::Mat matY1(matxyz1, cv::Rect(0, 1, matxyz1.cols, 1));
        cv::Mat matxyz1_Square = matxyz1.mul(matxyz1);
        cv::Mat matxyz1_Square_ROI(matxyz1_Square, cv::Rect(0, 0, matxyz1_Square.cols, 2));
        cv::Mat matR_2;
        cv::reduce(matxyz1_Square_ROI, matR_2, 0, cv::ReduceTypes::REDUCE_SUM);
        cv::Mat matR_4 = matR_2.mul(matR_2);
        cv::Mat matR_6 = matR_4.mul(matR_2);

        double k1 = stRpy.matDistCoeffs.at<double>(0, 0);
        double k2 = stRpy.matDistCoeffs.at<double>(0, 1);
        double p1 = stRpy.matDistCoeffs.at<double>(0, 2);
        double p2 = stRpy.matDistCoeffs.at<double>(0, 3);
        double k3 = stRpy.matDistCoeffs.at<double>(0, 4);

        cv::Mat matFactor = cv::Mat::ones(1, matR_2.cols, matR_2.type()) + k1 * matR_2 + k2 * matR_4 + k3 * matR_6;

        cv::Mat matX2 = matX1.mul(matFactor) + 2. * p1 * matX1.mul(matY1) + p2 * (matR_2 + 2. * matX1.mul(matX1));
        cv::Mat matY2 = matY1.mul(matFactor) + p1 * (matR_2 + 2. * matY1.mul(matY1)) + 2. * p2 * matX1.mul(matY1);

        cv::Mat matXYZ_2 = matX2.clone();
        matXYZ_2.push_back(matY2);
        cv::Mat matZ_2 = cv::Mat::ones(1, matXYZ_2.cols, matXYZ_2.type());
        matXYZ_2.push_back(matZ_2);
        cv::Mat matUV = stRpy.matIntrinsicMatrix * matXYZ_2;
        auto vecVecProjectPoints = matToVector<double>(matUV);

        std::ofstream ofStrm;
        ofStrm.open("./data/CoordinateDiffFinalMatrix.txt");
        for (size_t index = 0; index < stRpy.vecImagePoints.size(); ++index) {
            cv::Point2f ptImage = stRpy.vecImagePoints[index];
            cv::Point2f ptProject = cv::Point2f ( vecVecProjectPoints[0][index], vecVecProjectPoints[1][index]);
            ofStrm << ptImage.x - ptProject.x << ", " << ptImage.y - ptProject.y << std::endl;
        }
        ofStrm.close();
    }

    {
        cv::Mat matProject = stRpy.matInitialIntrinsicMatrix * stRpy.matInitialExtrinsicMatrix * matPoint;
        double dScale = matProject.at<double>(2, 0);
        matProject = matProject / dScale;
        auto vecVecProjectPoints = matToVector<double>(matProject);

        std::ofstream ofStrm;
        ofStrm.open("./data/CoordinateDiffInitialMatrix.txt");
        for (size_t index = 0; index < stRpy.vecImagePoints.size(); ++index) {
            cv::Point2f ptImage = stRpy.vecImagePoints[index];
            cv::Point2f ptProject = cv::Point2f ( vecVecProjectPoints[0][index], vecVecProjectPoints[1][index] );
            ofStrm << ptImage.x - ptProject.x << ", " << ptImage.y - ptProject.y << std::endl;
        }
        ofStrm.close();
    }
    //PR_RESTORE_IMG_CMD stRetoreCmd;
    //PR_RESTORE_IMG_RPY stRetoreRpy;
    //stRetoreCmd.matInputImg = cv::imread(strImgPath, cv::IMREAD_GRAYSCALE );
    //stRetoreCmd.vecMatRestoreImage = stRpy.vecMatRestoreImage; //stCalcUndistortRectifyMapRpy.vecMatRestoreImage;
    //PR_RestoreImage ( &stRetoreCmd, &stRetoreRpy );
    //if ( VisionStatus::OK != stRetoreRpy.enStatus) {
    //    std::cout << "Failed to restore image!" << std::endl;
    //    return;
    //}

    //cv::Mat matInputFloat;
    //stRetoreCmd.matInputImg.convertTo ( matInputFloat, CV_32FC1 );

    ////cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - stRetoreRpy.matResultImg );
    //cv::imwrite("./data/chessboardRestored.png", stRetoreRpy.matResultImg);

    //cv::Mat matReadRestored = cv::imread("./data/chessboardRestored.png", cv::IMREAD_GRAYSCALE );
    //cv::Mat matReadAgainFloat;
    //matReadRestored.convertTo ( matReadAgainFloat, CV_32FC1 );

    //cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - matReadAgainFloat );
    //cv::imwrite("./data/chessboard_diff.png", matDiff);
    PR_DumpTimeLog("./data/TimeLog.log");
}


void TestCompareInputAndResult() {
    cv::Mat matInputImg = cv::imread ( "./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093/image.png");
    if ( matInputImg.empty() )
        return;

    cv::Mat matResultImg = cv::imread ( "./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093/result.png" );
    if ( matResultImg.empty() )
        return;

    cv::Mat matInputFloat, matResultFloat;
    matInputImg.convertTo  ( matInputFloat, CV_32FC1 );
    matResultImg.convertTo ( matResultFloat, CV_32FC1 );

    cv::Mat matDiff = cv::Scalar::all(255.f) - ( matInputFloat - matResultFloat );
    cv::imwrite("./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093/chessboard_diff.png", matDiff);
}

void TestRunRestoreImgLogCase() {
    PR_RunLogCase ("./Vision/Logcase/RestoreImg_2017_04_27_16_48_50_093");
}