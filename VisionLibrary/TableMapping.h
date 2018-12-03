#pragma once

#include "VisionHeader.h"

namespace AOI
{
namespace Vision
{

class TableMapping
{

    TableMapping();
    ~TableMapping();
    static cv::Mat _fitModel(const PR_TABLE_MAPPING_CMD::VectorOfFramePoint &framePoints, cv::Mat& matXX, cv::Mat& matYY);
    static PR_TABLE_MAPPING_CMD::VectorOfFramePoint _preHandleData(const PR_TABLE_MAPPING_CMD::VectorOfFramePoint &framePoints);
    static float _calcTransR1RSum(const PR_TABLE_MAPPING_CMD::VectorOfFramePoints& vecFramePoints,
        const cv::Mat& matD1, cv::Mat& matR1, cv::Mat& matDR1);    
    static int _findMatchPoint(const VectorOfFloat& vecTargetX, const VectorOfFloat& vecTargetY, const cv::Point2f& ptSrch);
    static cv::Mat _paraFromPolyNomial(int n);
    static cv::Mat _generateBazier(const cv::Mat& matX, const cv::Mat& matY, const VectorOfFloat& vecXyMinMax, int mm, int nn);
    static cv::Mat _calculatePPz(const cv::Mat& matX, const cv::Mat& matY, const cv::Mat& matZ,
        const VectorOfFloat& vecXyMinMax, int mm, int nn);
    static cv::Mat _calculateSurface(const cv::Mat& matX, const cv::Mat& matY, const cv::Mat& matPPz, const VectorOfFloat& vecXyMinMax, int mm, int nn);

public:
    static VisionStatus run(const PR_TABLE_MAPPING_CMD *const pstCmd, PR_TABLE_MAPPING_RPY *const pstRpy);
};

}
}
