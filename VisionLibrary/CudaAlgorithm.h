#pragma once

#include <cuda_runtime.h>
#include "opencv2/core.hpp"
#include "VisionHeader.h"
#include "opencv2/core/cuda.hpp"

namespace AOI
{
namespace Vision
{

struct Calc3DHeightVars {
    cv::cuda::GpuMat matAlpha;
    cv::cuda::GpuMat matBeta;
    cv::cuda::GpuMat matGamma;
    cv::cuda::GpuMat matGamma1;
    cv::cuda::GpuMat matAvgUnderTolIndex;
    cv::cuda::GpuMat matBufferGpu;
    cv::cuda::GpuMat matBufferGpuT;
    cv::cuda::GpuMat matBufferGpuT_1;
    cv::cuda::GpuMat matMaskGpu;
    cv::cuda::GpuMat matMaskGpuT;
    cv::cuda::GpuMat matMaskGpu_1;
    cv::cuda::GpuMat matMaskGpuT_1;
    cv::cuda::GpuMat matMaskGpu_2;
    cv::cuda::GpuMat matMaskGpuT_2;
    cv::cuda::GpuMat matDiffResult;
    cv::cuda::GpuMat matDiffResultT;
    cv::cuda::GpuMat matDiffResult_1;
    cv::cuda::GpuMat matDiffResultT_1;
    cv::cuda::GpuMat matBufferSign;     // Buffer to use in phase correction
    cv::cuda::GpuMat matBufferAmpl;     // Buffer to use in phase correction
    float* d_p3;
    float* d_tmpResult;
    int* pBufferJumpSpan;
    int* pBufferJumpStart;
    int* pBufferJumpEnd;
    int* pBufferSortedJumpSpanIdx;
    cudaEvent_t eventDone;
};

struct DlpCalibResult {
    cv::cuda::GpuMat matAlphaBase;
    cv::cuda::GpuMat matBetaBase;
    cv::cuda::GpuMat matGammaBase;
    float* pDlp3DBezierSurface;
};

class CudaAlgorithm
{
    CudaAlgorithm() = delete;
    ~CudaAlgorithm() = delete;
    static DlpCalibResult m_dlpCalibData[NUM_OF_DLP];
    static Calc3DHeightVars m_arrCalc3DHeightVars[NUM_OF_DLP];

public:
    static DlpCalibResult& getDlpCalibData(int nDlp) { return m_dlpCalibData[nDlp]; }
    static Calc3DHeightVars& getCalc3DHeightVars(int nDLp) { return m_arrCalc3DHeightVars[nDLp]; }
    static bool initCuda();
    static VisionStatus setDlpParams(const PR_SET_DLP_PARAMS_TO_GPU_CMD* const pstCmd, PR_SET_DLP_PARAMS_TO_GPU_RPY* const pstRpy);
    static cv::cuda::GpuMat diff(const cv::cuda::GpuMat& matInput, int nRecersiveTime, int nDimension, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void diff(const cv::cuda::GpuMat& matInput,
        cv::cuda::GpuMat& matResult,
        int nDimension,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void floor(cv::cuda::GpuMat &matInOut, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void calcPhase(
        const cv::cuda::GpuMat& matInput0,
        const cv::cuda::GpuMat& matInput1,
        const cv::cuda::GpuMat& matInput2,
        const cv::cuda::GpuMat& matInput3,
        cv::cuda::GpuMat& matPhase,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void calcPhaseAndMask(
        const cv::cuda::GpuMat& matInput0,
        const cv::cuda::GpuMat& matInput1,
        const cv::cuda::GpuMat& matInput2,
        const cv::cuda::GpuMat& matInput3,
        cv::cuda::GpuMat& matPhase,
        cv::cuda::GpuMat& matMask,
        float fMinimumAlpitudeSquare,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static float intervalAverage(const cv::cuda::GpuMat& matInput, int interval, float *d_result, cudaEvent_t& eventDone,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static float intervalRangeAverage(const cv::cuda::GpuMat& matInput, int interval, float rangeStart, float rangeEnd, float* d_result,
        cudaEvent_t& eventDone,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseWrapBuffer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matBuffer, float* d_tmpVar, float fShift,
        cudaEvent_t& eventDone,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseWrapByRefer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matRef, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseWarpNoAutoBase(const cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matResult, float fShift, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseCorrectionCmp(
        cv::cuda::GpuMat& matPhase,
        const cv::cuda::GpuMat& matPhase1,
        cv::cuda::GpuMat& matBufferGpu,
        cv::cuda::GpuMat& matBufferGpuT,
        cv::cuda::GpuMat& matDiffMapX,
        cv::cuda::GpuMat& matDiffMapY,
        cv::cuda::GpuMat& matDiffPhaseX,
        cv::cuda::GpuMat& matDiffPhaseY,
        cv::cuda::GpuMat& matMaskGpu,
        cv::cuda::GpuMat& matMaskGpuT,
        int span,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void phaseCorrection(
        cv::cuda::GpuMat& matPhase,
        cv::cuda::GpuMat& matPhaseT,
        cv::cuda::GpuMat& matDiffResult,
        cv::cuda::GpuMat& matDiffResultT,
        cv::cuda::GpuMat& matBufferSign,
        cv::cuda::GpuMat& matBufferAmpl,
        int* pBufferJumpSpan,
        int* pBufferJumpStart,
        int* pBufferJumpEnd,
        int* pBufferSortedJumpSpanIdx,
        int nJumpSpanX,
        int nJumpSpanY,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void phaseCorrection(cv::Mat &matPhase, int nJumpSpanX, int nJumpSpanY);

    static void calculateSurfaceConvert3D(
        const cv::cuda::GpuMat& matPhase,
        cv::cuda::GpuMat& matPhaseT,
        cv::cuda::GpuMat& matBufferGpu,
        const VectorOfFloat& param,
        const int ss,
        const int nDlpNo,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void phaseToHeight3D(
        const cv::cuda::GpuMat& zInLine,
        const cv::cuda::GpuMat& matPolyParassGpu,
        const float* d_xxt,
        const int xxtStep,
        const int len,
        const int ss,
        float* d_P3,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void calcSumAndConvertMatrix(
        float* d_pInputData,
        int ss,
        cv::cuda::GpuMat& matResult,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void medianFilter(
        const cv::cuda::GpuMat& matInput,
        cv::cuda::GpuMat& matOutput,
        const int windowSize,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void phasePatch(
        cv::cuda::GpuMat& matInOut,
        const cv::cuda::GpuMat& matNanMask,
        PR_DIRECTION enProjDir,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void getNanMask(
        cv::cuda::GpuMat& matInput,
        cv::cuda::GpuMat& matNanMask,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void mergeHeightIntersectCore(
        cv::cuda::GpuMat& matHGpu1,
        cv::cuda::GpuMat& matHGpu2,
        cv::cuda::GpuMat& matHGpu3,
        cv::cuda::GpuMat& matHGpu4,
        cv::cuda::GpuMat& matNanMask,
        cv::cuda::GpuMat& matNanMaskDiff,
        float fDiffThreshold,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void mergeHeightIntersect(
        cv::cuda::GpuMat& matHeightOneInput,
        cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat& matHeightTwoInput,
        cv::cuda::GpuMat& matNanMaskTwo,
        cv::cuda::GpuMat& matBufferGpu1,
        cv::cuda::GpuMat& matBufferGpu2,
        cv::cuda::GpuMat& matHeightOne,
        cv::cuda::GpuMat& matHeightTwo,
        cv::cuda::GpuMat& matHeightTre,
        cv::cuda::GpuMat& matHeightFor,
        cv::cuda::GpuMat& matMaskGpu1,
        cv::cuda::GpuMat& matMaskGpu2,
        cv::cuda::GpuMat& matMaskGpu3,
        cv::cuda::GpuMat& matMaskGpu4,
        cv::cuda::GpuMat& matAbsDiff, //Float
        cv::cuda::GpuMat& matBigDiffMask,
        cv::cuda::GpuMat& matBigDiffMaskFloat,
        cv::cuda::GpuMat& matDiffResultDiff,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::GpuMat& matMergeResult,
        cv::cuda::GpuMat& matResultNan,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static cv::cuda::GpuMat runMergeHeightIntersect(
        cv::cuda::GpuMat& matHeightOneInput,
        cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat& matHeightTwoInput,
        cv::cuda::GpuMat& matNanMaskTwo,
        Calc3DHeightVars& calc3DHeightVar0,
        Calc3DHeightVars& calc3DHeightVar1,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::GpuMat& matResultNan,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static void chooseMinValueForMask(
        cv::cuda::GpuMat&       matH1,
        cv::cuda::GpuMat&       matH2,
        cv::cuda::GpuMat&       matH3,
        const cv::cuda::GpuMat& matMask,
        cv::cuda::Stream&       stream = cv::cuda::Stream::Null());

    static void mergeHeightIntersect06(
        cv::cuda::GpuMat& matHeightOneInput,
        cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat& matHeightTwoInput,
        cv::cuda::GpuMat& matNanMaskTwo,
        cv::cuda::GpuMat& matBufferGpu1,
        cv::cuda::GpuMat& matBufferGpu2,
        cv::cuda::GpuMat& matHeightOne,
        cv::cuda::GpuMat& matHeightTwo,
        cv::cuda::GpuMat& matHeightTre,
        cv::cuda::GpuMat& matHeightFor,
        cv::cuda::GpuMat& matAbsDiff, //Float
        cv::cuda::GpuMat& matBigDiffMask,
        cv::cuda::GpuMat& matBigDiffMaskFloat,
        cv::cuda::GpuMat& matDiffResultDiff,
        float             fDiffThreshold,
        PR_DIRECTION      enProjDir,
        cv::cuda::GpuMat& matMergeResult,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());

    static cv::cuda::GpuMat runMergeHeightIntersect06(
        cv::cuda::GpuMat& matHeightOneInput,
        cv::cuda::GpuMat& matNanMaskOne,
        cv::cuda::GpuMat& matHeightTwoInput,
        cv::cuda::GpuMat& matNanMaskTwo,
        Calc3DHeightVars& calc3DHeightVar0,
        Calc3DHeightVars& calc3DHeightVar1,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
};

}
}
