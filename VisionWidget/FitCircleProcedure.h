#pragma once
#include "Procedure.h"

class FitCircleProcedure :
    public Procedure
{
public:
    FitCircleProcedure(VisionView *pVisionView);
    ~FitCircleProcedure();
    virtual int run(const std::string &imagePath) override;
    void setErrTol(float fErrTol);
    void setThreshold(int nThreshold);
    void setAlgorithm( int nAlgorithm);
    void setAttribute(int nAttribute);
    void setDirection(int nDirection);
    cv::Mat getResultMat() const;
private:
    cv::Point                       _ptCircleCtr;
    float                           _fInnerRangeRadius;
    float                           _fOutterRangeRadius;
    float                           _fErrorTol;
    int                             _nThreshold;
    int                             _nAlgorithm;
    int                             _nAttribute;
    int                             _nDirection;
    cv::Mat                         _matResult;
    VisionStatus fitCircle(const std::string &imagePath);
};