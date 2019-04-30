#include <cuda_runtime.h>

#include "CudaAlgorithm.h"
#include "../VisionLibrary/StopWatch.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/highgui.hpp"
#include "CudaFunc.h"
#include "TimeLog.h"
#include "CalcUtils.hpp"
#include "CudaAlgorithm.h"

namespace AOI
{
namespace Vision
{

/*static*/ void CudaAlgorithm::phaseCorrectionCmp(cv::cuda::GpuMat& matPhase, const cv::cuda::GpuMat& matPhase1, int span) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    static int time = 0;
    ++ time;

    cv::cuda::GpuMat matIdx;
    cv::cuda::compare(matPhase, matPhase1, matIdx, cv::CmpTypes::CMP_NE);
    cv::cuda::GpuMat matMap(ROWS, COLS, CV_32FC1);
    matMap.setTo(1, matIdx);

    // Note even the size of Mat ix ROW * (COL - 1), but actually the background memory size is still ROW * COL
    cv::cuda::GpuMat dMap1 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::cuda::GpuMat dxPhase = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);

    cv::cuda::GpuMat idxl1(ROWS, COLS, CV_8UC1);
    idxl1.setTo(0);

    TimeLog::GetInstance()->addTimeLog("prepare data for phaseCorrectionCmp X", stopWatch.Span());

    // Select in X direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap1.data),
        reinterpret_cast<float *>(dxPhase.data),
        idxl1.data,
        dMap1.step1(),
        ROWS, COLS, span);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point X", stopWatch.Span());

    // Select in Y direction, transpose the matrix to accelerate
    cv::cuda::transpose(matMap, matMap);
    cv::cuda::GpuMat matPhaseT;
    cv::cuda::transpose(matPhase, matPhaseT);

    cv::cuda::GpuMat dMap2 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::cuda::GpuMat dyPhase = CalcUtils::diff(matPhaseT, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::cuda::GpuMat idxl2(COLS, ROWS, CV_8UC1);    // Note the indl2 is tranposed result, so the ROWS and COLS are changed
    idxl2.setTo(0);

    TimeLog::GetInstance()->addTimeLog("prepare data for phaseCorrectionCmp Y", stopWatch.Span());

    // Select in Y direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap2.data),
        reinterpret_cast<float *>(dyPhase.data),
        idxl2.data,
        dMap2.step1(),
        COLS, ROWS, span);

    cv::cuda::transpose(idxl2, idxl2);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point Y", stopWatch.Span());

    cv::cuda::bitwise_and(idxl1, idxl2, idxl1);
    matPhase1.copyTo(matPhase, idxl1);

    TimeLog::GetInstance()->addTimeLog("bitwise_and and copy data", stopWatch.Span());
}

/*static*/ void CudaAlgorithm::phaseCorrectionCmp(cv::Mat& matPhase, const cv::Mat& matPhase1, int span) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;
    static int time = 0;
    ++ time;

    cv::Mat matIdx;
    cv::compare(matPhase, matPhase1, matIdx, cv::CmpTypes::CMP_NE);
    cv::Mat matMap = cv::Mat::zeros(ROWS, COLS, CV_32FC1);
    matMap.setTo(1, matIdx);

    // Note even the size of Mat ix ROW * (COL - 1), but actually the background memory size is still ROW * COL
    cv::Mat dMap1 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::Mat dxPhase = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);

    if (time == 1) {
        CalcUtils::saveMatToCsv(dMap1, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDMap1Cpu.csv");
        CalcUtils::saveMatToCsv(dxPhase, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDXPhaseCpu.csv");
    }

    cv::Mat idxl1(ROWS, COLS, CV_8UC1);
    idxl1.setTo(0);

    TimeLog::GetInstance()->addTimeLog("prepare for phaseCorrectionCmp X", stopWatch.Span());

    // Select in X direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap1.data),
        reinterpret_cast<float *>(dxPhase.data),
        idxl1.data,
        dMap1.step1(),
        ROWS, COLS, span);
    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point X", stopWatch.Span());

    cv::imwrite(std::string("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpMask1_") + std::to_string(time) + ".png", idxl1);
    int nonZeroCount1 = cv::countNonZero(idxl1);

    // Select in Y direction, transpose the matrix to accelerate
    cv::Mat matMapT;
    cv::transpose(matMap, matMapT);
    cv::Mat matPhaseT;
    cv::transpose(matPhase, matPhaseT);

    cv::Mat dMap2 = CalcUtils::diff(matMapT, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::Mat dyPhase = CalcUtils::diff(matPhaseT, 1, CalcUtils::DIFF_ON_X_DIR);
    cv::Mat idxl2(COLS, ROWS, CV_8UC1);    // Note the indl2 is tranposed result, so the ROWS and COLS are changed
    idxl2.setTo(0);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point Y", stopWatch.Span());

    if (time == 1) {
        CalcUtils::saveMatToCsv(dMap2, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDMap2.csv");
        CalcUtils::saveMatToCsv(dyPhase, "C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpDYPhase.csv");
    }

    // Select in Y direction
    run_kernel_select_cmp_point(2, 256,
        reinterpret_cast<float *>(dMap2.data),
        reinterpret_cast<float *>(dyPhase.data),
        idxl2.data,
        dMap2.step1(),
        COLS, ROWS, span);

    cv::imwrite(std::string("C:/Data/3D_20190408/PCBFOV20190104/Frame_1_Result/PhaseCorrectionCmpMask2_") + std::to_string(time) + ".png", idxl2);
    int nonZeroCount2 = cv::countNonZero(idxl2);

    cv::transpose(idxl2, idxl2);

    TimeLog::GetInstance()->addTimeLog("run_kernel_select_cmp_point Y", stopWatch.Span());

    cv::bitwise_and(idxl1, idxl2, idxl1);
    int nonZeroFinal = cv::countNonZero(idxl1);

    std::cout << "phaseCorrectionCmp point count1 " << nonZeroCount1 << " count2 " << nonZeroCount2 << " final cout " << nonZeroFinal << std::endl;

    matPhase1.copyTo(matPhase, idxl1);
}

/*static*/ cv::Mat CudaAlgorithm::mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir) {
    CStopWatch stopWatch;
    auto start = stopWatch.AbsNowInMicro();

    matHeightTwo.copyTo(matHeightOne, matNanMaskOne);
    matHeightOne.copyTo(matHeightTwo, matNanMaskTwo);

    if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
        cv::transpose(matHeightOne, matHeightOne);
        cv::transpose(matHeightTwo, matHeightTwo);

        TimeLog::GetInstance()->addTimeLog("_mergeHeightIntersect. Transpose image.", stopWatch.Span());
    }

    //#ifdef _DEBUG
    //    auto vecVecHeightOne = matToVector<float>(matHeightOne);
    //    auto vecVecHeightTwo = matToVector<float>(matHeightTwo);
    //    auto vecVecHeightTre = matToVector<float>(matHeightTre);
    //    auto vecVecHeightFor = matToVector<float>(matHeightFor);
    //#endif

    std::cout << "Prepare data " << stopWatch.SpanInMicro() << std::endl;

    //matHeightOne = matHeightOne.reshape(1, 1);
    //matHeightTwo = matHeightTwo.reshape(1, 1);
    //matHeightTre = matHeightTre.reshape(1, 1);
    //matHeightFor = matHeightFor.reshape(1, 1);
    cv::cuda::GpuMat matHeightOneGpu, matHeightTwoGpu;
    //float *d_result = NULL;
    //cudaMalloc(&d_result, matHeightOne.step * matHeightOne.rows);
    //auto size = matHeightOne.step * matHeightOne.rows;

    //float *d_One = NULL, *d_Two = NULL, *d_Tre = NULL, *d_For = NULL, *d_Mask = NULL;

    //int status = cudaHostRegister(matHeightOne.data, size, cudaHostRegisterPortable);
    cv::cuda::registerPageLocked(matHeightOne);
    std::cout << "Register matrix one " << stopWatch.SpanInMicro() << std::endl;

    matHeightOneGpu.upload(matHeightOne);
    //printf("Copy input data from the host memory to the CUDA device\n");
    //err = cudaMemcpy(d_One, matHeightOne.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector one from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    std::cout << "Upload matrix one " << stopWatch.SpanInMicro() << std::endl;

    cv::cuda::registerPageLocked(matHeightTwo);
    //cudaHostRegister(matHeightTwo.data, matHeightTwo.step * matHeightTwo.rows, cudaHostRegisterPortable);
    std::cout << "Register matrix two " << stopWatch.SpanInMicro() << std::endl;

    matHeightTwoGpu.upload(matHeightTwo);
    //err = cudaMemcpy(d_Two, matHeightTwo.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector two from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    std::cout << "Upload matrix two " << stopWatch.SpanInMicro() << std::endl;

    //cv::cuda::registerPageLocked(matHeightTre); matHeightTreGpu.upload(matHeightTre);
    //cv::cuda::registerPageLocked(matNanMask);   matNanMaskGpu.upload(matNanMask);

    //cv::cuda::registerPageLocked(matHeightFor); matHeightForGpu.upload(matHeightFor);

    //err = cudaMemcpy(d_Tre, matHeightTre.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector three from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}

    //err = cudaMemcpy(d_For, matHeightTre.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector four from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}

    //err = cudaMemcpy(d_Mask, matNanMask.data, size, cudaMemcpyHostToDevice);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector four from host to device (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}
    // matHeightTwoGpu.upload(matHeightTwo);
    // matHeightTreGpu.upload(matHeightTre);
    // matHeightForGpu.upload(matHeightFor);
    //matNanMaskGpu.upload(matNanMask);

    cv::cuda::GpuMat matHeightForGpu(matHeightOne.size(), matHeightOne.type());

    std::cout << "Upload data " << stopWatch.SpanInMicro() << std::endl;
    std::cout << "rows " << matHeightOne.rows << " cols " << matHeightOne.cols << std::endl;
    int gnBlockNumber = 2;
    int gnThreadNumber = 256;
    run_kernel_merge_height_intersect(
        gnBlockNumber,
        gnThreadNumber,
        reinterpret_cast<float *>(matHeightOneGpu.data),
        reinterpret_cast<float *>(matHeightTwoGpu.data),
        reinterpret_cast<float *>(matHeightForGpu.data),
        matHeightOne.rows,
        matHeightOne.cols,
        fDiffThreshold);

    std::cout << "kernel_merge_height " << stopWatch.SpanInMicro() << std::endl;

    cv::Mat matResult;
    matHeightForGpu.download(matResult);
    //err = cudaMemcpy(matHeightFor.data, d_For, size, cudaMemcpyDeviceToHost);
    //if (err != cudaSuccess)
    //{
    //    fprintf(stderr, "Failed to copy vector four from device to host (error code %s)!\n", cudaGetErrorString(err));
    //    exit(EXIT_FAILURE);
    //}

    std::cout << "Download data " << stopWatch.SpanInMicro() << std::endl;

    cv::cuda::unregisterPageLocked(matHeightOne);
    cv::cuda::unregisterPageLocked(matHeightTwo);

    //if (PR_DIRECTION::UP == enProjDir || PR_DIRECTION::DOWN == enProjDir) {
    //    cv::transpose(matHeightFor, matHeightFor);
    //    TimeLog::GetInstance()->addTimeLog("_mergeHeightIntersect. Transpose image back.", stopWatch.Span());
    //}

    auto total = stopWatch.AbsNowInMicro() - start;
    std::cout << "Total time: " << total << std::endl;

    return matResult;
}

}
}