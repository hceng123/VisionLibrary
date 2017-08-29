#ifndef _AOI_UNWRAP_H_
#define _AOI_UNWRAP_H_

#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

using DATA_TYPE = float;

class Unwrap
{
public:
    Unwrap ();
    ~Unwrap ();
    static void calib3DBase(const std::vector<cv::Mat> &vecInputImgs, bool bGuassianFilter, bool bReverseSeq, cv::Mat &matK, cv::Mat &matPPz );
    static void calc3DHeight(const std::vector<cv::Mat> &vecInputImgs, bool bGuassianFilter, bool bReverseSeq, const cv::Mat &matK, const cv::Mat &matPPz, cv::Mat &matHeight);
private:
    static cv::Mat getResidualPoint(const cv::Mat &matInput, cv::Mat &matPosPole, cv::Mat &matNegPole);
    static VectorOfPoint selectLonePole(ListOfPoint &listPoint, int width, int height, size_t nLonePoneCount );
    static void selectPolePair(const VectorOfPoint &vecPosPoint, const VectorOfPoint &vecNegPoint, int rows, int cols, VectorOfPoint &vecResult1, VectorOfPoint &vecResult2, VectorOfPoint &vecResult3 );
    static VectorOfPoint getIntegralTree(const VectorOfPoint &vecResult1, const VectorOfPoint &vecResult2, const VectorOfPoint &vecResult3 );
    static cv::Mat phaseUnwrapSurface(const cv::Mat &matPhase);
    static cv::Mat phaseUnwrapSurfaceTrk ( const cv::Mat &matPhase, const cv::Mat &matBranchCut);
    static cv::Mat phaseUnwrapSurfaceByRefer(const cv::Mat &matPhase, float Thresh, const cv::Mat &matRef );
    static cv::Mat calculatePPz(const cv::Mat &matX, const cv::Mat &matY, const cv::Mat &matZ);
    static cv::Mat calculateSurface(const cv::Mat &matZ, const cv::Mat &matPPz);
    static cv::Mat _findUnstablePoint(const std::vector<cv::Mat> &vecInputImgs);

    static const int GUASSIAN_FILTER_SIZE =  5;
    static const float GUASSIAN_FILTER_SIGMA;
    static const float ONE_HALF_CYLE;
    static const int BEZIER_RANK = 5;
    static const float UNSTABLE_DIFF;
};

}
}
#endif /*_AOI_UNWRAP_H_*/