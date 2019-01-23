#ifndef _AOI_UNWRAP_H_
#define _AOI_UNWRAP_H_

#include "VisionHeader.h"
#include "CalcUtils.hpp"
#include "TimeLog.h"

namespace AOI
{
namespace Vision
{

using DATA_TYPE = float;

//This is to get the same result as matlab, so easy to bench mark the result with matlab.
//The color sequence get from https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html
static const cv::Scalar MATLAB_COLOR_ORDER[] = {
    cv::Scalar(0.7410 * PR_MAX_GRAY_LEVEL, 0.4470 * PR_MAX_GRAY_LEVEL, 0.0000 * PR_MAX_GRAY_LEVEL),
    cv::Scalar(0.0980 * PR_MAX_GRAY_LEVEL, 0.3250 * PR_MAX_GRAY_LEVEL, 0.8500 * PR_MAX_GRAY_LEVEL),
    cv::Scalar(0.1250 * PR_MAX_GRAY_LEVEL, 0.6940 * PR_MAX_GRAY_LEVEL, 0.9290 * PR_MAX_GRAY_LEVEL),
    cv::Scalar(0.5560 * PR_MAX_GRAY_LEVEL, 0.1840 * PR_MAX_GRAY_LEVEL, 0.4940 * PR_MAX_GRAY_LEVEL),
    cv::Scalar(0.1880 * PR_MAX_GRAY_LEVEL, 0.6740 * PR_MAX_GRAY_LEVEL, 0.4660 * PR_MAX_GRAY_LEVEL),
    cv::Scalar(0.9330 * PR_MAX_GRAY_LEVEL, 0.7450 * PR_MAX_GRAY_LEVEL, 0.3010 * PR_MAX_GRAY_LEVEL),
    cv::Scalar(0.1840 * PR_MAX_GRAY_LEVEL, 0.0780 * PR_MAX_GRAY_LEVEL, 0.6350 * PR_MAX_GRAY_LEVEL),
};
const static int MATLAB_COLOR_COUNT = 7;
static_assert ( sizeof ( MATLAB_COLOR_ORDER ) / sizeof ( cv::Scalar ) == MATLAB_COLOR_COUNT, "The size of MATLAB_COLOR_COUNT is not correct" );

class Unwrap
{
public:
    Unwrap ();
    ~Unwrap ();
    static void initAtan2Table();
    static void calib3DBase(const PR_CALIB_3D_BASE_CMD *const pstCmd, PR_CALIB_3D_BASE_RPY *const pstRpy);
    static cv::Mat calculateBaseSurface(int rows, int cols, const cv::Mat &matBaseSurfaceParam);
    static void calib3DHeight(const PR_CALIB_3D_HEIGHT_CMD *const pstCmd, PR_CALIB_3D_HEIGHT_RPY *const pstRpy);
    static void integrate3DCalib(const PR_INTEGRATE_3D_CALIB_CMD *const pstCmd, PR_INTEGRATE_3D_CALIB_RPY *const pstRpy);
    static void motorCalib3D(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_RPY *const pstRpy);
    static void motorCalib3DNew(const PR_MOTOR_CALIB_3D_CMD *const pstCmd, PR_MOTOR_CALIB_3D_RPY *const pstRpy);
    static void calc3DHeight(const PR_CALC_3D_HEIGHT_CMD *const pstCmd, PR_CALC_3D_HEIGHT_RPY *const pstRpy);
    static void merge3DHeight(const PR_MERGE_3D_HEIGHT_CMD *const pstCmd, PR_MERGE_3D_HEIGHT_RPY *const pstRpy);
    static void calc3DHeightDiff(const PR_CALC_3D_HEIGHT_DIFF_CMD *const pstCmd, PR_CALC_3D_HEIGHT_DIFF_RPY *const pstRpy);
    static void calcMTF(const PR_CALC_MTF_CMD *const pstCmd, PR_CALC_MTF_RPY *const pstRpy);
    static void calcPD(const PR_CALC_PD_CMD *const pstCmd, PR_CALC_PD_RPY *const pstRpy);
    static void phaseCorrection(cv::Mat &matPhase, const cv::Mat &matIdxNan, int nJumpSpanX, int nJumpSpanY);   //Put it to public to include it in regression test.
    static void phaseCorrectionEx(cv::Mat &matPhase, PR_DIRECTION enProjDir, PR_DIRECTION enScanDir, int nMinSpan, int nMaxSpan);
    static void removeJumpArea(cv::Mat &matHeight, float fAreaLimit);
    static void insp3DSolder(const PR_INSP_3D_SOLDER_CMD *pstCmd, const cv::Mat &matBaseMask, PR_INSP_3D_SOLDER_RPY *const pstRpy);

private:
    static cv::Mat _getResidualPoint(const cv::Mat &matPhaseDx, const cv::Mat &matPhaseDy, cv::Mat &matPosPole, cv::Mat &matNegPole);
    static VectorOfPoint _selectLonePole(ListOfPoint &listPoint, int width, int height, size_t nLonePoneCount);
    static void _selectPolePair(const VectorOfPoint &vecPosPoint, const VectorOfPoint &vecNegPoint, int rows, int cols, VectorOfPoint &vecResult1, VectorOfPoint &vecResult2, VectorOfPoint &vecResult3 );
    static VectorOfPoint _getIntegralTree(const VectorOfPoint &vecResult1, const VectorOfPoint &vecResult2, const VectorOfPoint &vecResult3, int rows, int cols );
    static cv::Mat _phaseUnwrapSurface(const cv::Mat &matPhase);
    static cv::Mat _phaseUnwrapSurfaceTrk(const cv::Mat &matPhase, const cv::Mat &matPhaseDx, const cv::Mat &matPhaseDy, const cv::Mat &matBranchCut);
    static cv::Mat _phaseUnwrapSurfaceTrkNew(const cv::Mat &matPhase, const cv::Mat &matPhaseDx, const cv::Mat &matPhaseDy, const cv::Mat &matBranchCut);
    static cv::Mat _phaseUnwrapSurfaceByRefer(const cv::Mat &matPhase, const cv::Mat &matRef, const cv::Mat &matNan);
    static cv::Mat _calculatePPz(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matZ);
    static void _setBySign(cv::Mat &matInOut, DATA_TYPE value);
    static void _findUnstablePoint(const std::vector<cv::Mat> &vecInputImgs, float fDiffTol, float fAvgTol, cv::Mat &matDiffUnderTolIndex, cv::Mat &matAvgUnderTolIndex);
    static cv::Mat _drawHeightGrid(const cv::Mat &matHeight, int nGridRow, int nGridCol, const cv::Size &szMeasureWinSize, VectorOfFloat &vecGridHeights);
    static void _drawStar(cv::Mat &matInOut, cv::Point ptPosition, int nSize);
    static cv::Mat _calcHeightFromPhase(const cv::Mat &matPhase, const cv::Mat &matHtt, const cv::Mat &matK);
    static cv::Mat _calcHeightFromPhaseBuffer(const cv::Mat &matPhase, const cv::Mat &matHtt, const cv::Mat &matK, cv::Mat &matBuffer1, cv::Mat &matBuffer2);
    static cv::Mat _drawAutoThresholdResult(const cv::Mat &matInput, const VectorOfFloat &vecRanges);
    static void _fitHoleInNanMask(cv::Mat &matMask, const cv::Size &szMorphKernel, Int16 nMorphIteration);
    static void _phasePatch(cv::Mat &matPhase, PR_DIRECTION enProjDir, const cv::Mat &matNanMask);
    static void _phaseSwitch(cv::Mat &matPhase, PR_DIRECTION enProjDir, PR_DIRECTION enScanDir);
    static void _phaseSwitchBack(cv::Mat &matPhase, PR_DIRECTION enProjDir, PR_DIRECTION enScanDir);
    static void _calcReverseAndProjDir(const VectorOfMat &vecMat, bool bEnableGaussianFilter, PR_CALIB_3D_BASE_RPY *const pstRpy);
    static void _turnPhase(cv::Mat &matPhase, cv::Mat &matPhaseDiff, char *ptrSignOfRow, char *ptrAmplOfRow, int row, int nStart, int nEnd);
    static void _turnPhaseConditionOne(cv::Mat &matPhase, cv::Mat &matPhaseDiff, cv::Mat &matDiffSign, cv::Mat &matDiffAmpl, int row, int Nt1);
    static cv::Mat _mergeHeightIntersect(cv::Mat &matHeightOne, const cv::Mat &matNanMaskOne, cv::Mat &matHeightTwo, const cv::Mat &matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir);
    static cv::Mat _mergeHeightMax(cv::Mat &matHeightOne, const cv::Mat &matNanMaskOne, cv::Mat &matHeightTwo, const cv::Mat &matNanMaskTwo, float fDiffThreshold, PR_DIRECTION enProjDir);
    static bool _extractSolder(const cv::Mat &matHeightROI, float fCoverage, VectorOfFloat &vecThreshold, cv::Mat &matHighMask, cv::Mat &matMidMask, cv::Mat &matLowMask, int &nTopY, int &nBtmY);
    static float _calcSolderHeightXSG(const cv::Mat &matCheckROI, const cv::Mat &matMidMask);
    static float _calcSolderHeightTri(const cv::Mat &matCheckROI, const cv::Mat &matMidMask, int nWettingWidth, bool bLeft, int nTopY, int nBtmY, cv::Rect &rectCalc);
    static void _calculatePPzConvert3D(const cv::Mat& z1, const cv::Mat& matH, const VectorOfFloat& param, int ss, const cv::Mat& xxt, cv::Mat& matSum1t, cv::Mat& matSum2t);
    static cv::Mat _calculateSurfaceConvert3D(const cv::Mat& z1, const VectorOfFloat& param, int ss, const cv::Mat& xxt1);
    static inline void _phaseWrap(cv::Mat &matPhase) {
        CStopWatch stopWatch;
        cv::Mat matNn = matPhase / ONE_CYCLE + 0.5;
        TimeLog::GetInstance()->addTimeLog( "_phaseWrap Divide and add.", stopWatch.Span());

        CalcUtils::floorByRef<DATA_TYPE> ( matNn ); //The floor takes about 36ms, need to optimize

        TimeLog::GetInstance()->addTimeLog( "_phaseWrap floor takes.", stopWatch.Span() );

        matPhase = matPhase - matNn * ONE_CYCLE;

        TimeLog::GetInstance()->addTimeLog( "_phaseWrap multiply and subtract takes.", stopWatch.Span());
    }

    static inline void _phaseWrapBuffer(cv::Mat &matPhase, cv::Mat &matBuffer, float fShift = 0.f) {
        CStopWatch stopWatch;
        matBuffer = matPhase / ONE_CYCLE + 0.5 - fShift;
        TimeLog::GetInstance()->addTimeLog( "_phaseWrap Divide and add.", stopWatch.Span());

        CalcUtils::floorByRef<DATA_TYPE>(matBuffer);

        TimeLog::GetInstance()->addTimeLog("_phaseWrap floor takes.", stopWatch.Span());

        matBuffer = matPhase - matBuffer * ONE_CYCLE;

        TimeLog::GetInstance()->addTimeLog("_phaseWrap multiply and subtract takes.", stopWatch.Span());
        float fSum = 0.f, fCount = 0.f;
        for (int row = 0; row < matBuffer.rows; row += 20)
        for (int col = 0; col < matBuffer.cols; col += 20) {
            float value = matBuffer.at<DATA_TYPE>(row, col);
            if (value == value) {
                fSum += value;
                ++fCount;
            }
        }
        float fMean = fSum / fCount;

        matBuffer = matPhase / ONE_CYCLE + 0.5 - fShift - fMean / 2.f;
        CalcUtils::floorByRef<DATA_TYPE>(matBuffer);

        TimeLog::GetInstance()->addTimeLog("_phaseWrap floor takes.", stopWatch.Span());

        matPhase = matPhase - matBuffer * ONE_CYCLE;
    }

    static inline void _phaseWrapByRefer(cv::Mat &matPhase, cv::Mat &matRef) {
        CStopWatch stopWatch;
        matRef = (matPhase - matRef) / ONE_CYCLE + 0.5;  //Use matResult to store the substract result to reduce memory allocate time.
        TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer Divide and add.", stopWatch.Span());

        CalcUtils::floorByRef<DATA_TYPE>(matRef);
        TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer floor takes.", stopWatch.Span());

        matPhase = matPhase - matRef * ONE_CYCLE;
        TimeLog::GetInstance()->addTimeLog("_phaseWrapByRefer multiply and subtract takes.", stopWatch.Span());
    }

    static const int GUASSIAN_FILTER_SIZE   = 11;
    static const int BEZIER_RANK            = 5;
    static const int ERODE_WIN_SIZE         = 31;
    static const int PHASE_SNOOP_WIN_SIZE   = 10;
    static const int CALIB_HEIGHT_MIN_SIZE  = 200;
    static const int MEDIAN_FILTER_SIZE     = 5;
    static const int CHECK_RS_WIN           = 50;

    static const float GAUSSIAN_FILTER_SIGMA;
    static const float ONE_HALF_CYCLE;
    static const float ONE_CYCLE;
    static const float CALIB_HEIGHT_STEP_USEFUL_PT;
    static const float LOW_BASE_PHASE;
    static VectorOfVectorOfFloat vecVecAtan2Table;
};

}
}
#endif /*_AOI_UNWRAP_H_*/