#include "CudaAlgorithm.h"
#include "../VisionLibrary/StopWatch.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "CudaFunc.h"
#include "TimeLog.h"
#include "CalcUtils.hpp"

namespace AOI
{
namespace Vision
{

void phaseCorrectionCmp(cv::cuda::GpuMat& matPhase, const cv::cuda::GpuMat& matPhase1, int span) {
    CStopWatch stopWatch;
    const int ROWS = matPhase.rows;
    const int COLS = matPhase.cols;

    cv::Mat matIdx;
    cv::cuda::compare(matPhase, matPhase1, matIdx, cv::CmpTypes::CMP_NE);
    cv::Mat matMapCpu = cv::Mat::zeros(ROWS, COLS, CV_32FC1);
    cv::cuda::GpuMat matMap(matMapCpu);
    matMap.setTo(1, matIdx);

    auto dMap1 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_X_DIR);
    auto dMap2 = CalcUtils::diff(matMap, 1, CalcUtils::DIFF_ON_Y_DIR);

    auto dxPhase = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_X_DIR);
    auto dyPhase = CalcUtils::diff(matPhase, 1, CalcUtils::DIFF_ON_Y_DIR);

    auto idxl1 = cv::Mat::zeros(ROWS, COLS, CV_8UC1);
    auto idxl2 = cv::Mat::zeros(ROWS, COLS, CV_8UC1);

    TimeLog::GetInstance()->addTimeLog("prepare for phaseCorrectionCmp", stopWatch.Span());

    TimeLog::GetInstance()->addTimeLog("Finish search for position need to change", stopWatch.Span());

    matIdx = idxl1 & idxl2;
#ifdef _DEBUG
    auto vecVecIdx = CalcUtils::matToVector<uchar>(matIdx);
#endif
    matPhase1.copyTo(matPhase, matIdx);
}

cv::Mat CudaAlgorithm::mergeHeightIntersect(cv::Mat matHeightOne, cv::Mat matNanMaskOne, cv::Mat matHeightTwo, cv::Mat matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir) {
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