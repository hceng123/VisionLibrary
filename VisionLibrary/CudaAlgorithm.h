#pragma once
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
    cv::cuda::GpuMat matMaskGpu;
    cv::cuda::GpuMat matMaskGpuT;
    float* d_p3;
    float* d_tmpResult;
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
    static float intervalAverage(const cv::cuda::GpuMat& matInput, int interval, float *d_result, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static float intervalRangeAverage(const cv::cuda::GpuMat& matInput, int interval, float rangeStart, float rangeEnd, float* d_result,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseWrapBuffer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matBuffer, float* d_tmpVar, float fShift = 0.f, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseWrapByRefer(cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matRef, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseWarpNoAutoBase(const cv::cuda::GpuMat& matPhase, cv::cuda::GpuMat& matResult, float fShift, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseCorrectionCmp(cv::cuda::GpuMat& matPhase,
        const cv::cuda::GpuMat& matPhase1,
        cv::cuda::GpuMat& matBufferGpu,
        cv::cuda::GpuMat& matBufferGpuT,
        cv::cuda::GpuMat& matMaskGpu,
        cv::cuda::GpuMat& matMaskGpuT,
        int span,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
    static void phaseCorrectionCmp(cv::Mat& matPhase, const cv::Mat& matPhase1, int span);
    static void phaseCorrection(cv::cuda::GpuMat &matPhase, cv::cuda::GpuMat& matPhaseT, int nJumpSpanX, int nJumpSpanY, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
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

    static cv::Mat mergeHeightIntersect(cv::Mat matHeightOne,
        cv::Mat matNanMaskOne,
        cv::Mat matHeightTwo,
        cv::Mat matNanMaskTwo,
        float fDiffThreshold,
        PR_DIRECTION enProjDir,
        cv::cuda::Stream& stream = cv::cuda::Stream::Null());
};

}
}
