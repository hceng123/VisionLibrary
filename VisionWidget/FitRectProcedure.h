#pragma once
#include "Procedure.h"

class FitRectProcedure :
    public Procedure
{
public:
    FitRectProcedure(VisionView *pVisionView);
    ~FitRectProcedure();
    virtual int run(const std::string &imagePath) override;
    void setErrTol(float fErrTol);
    void setThreshold(int nThreshold);
    void setAlgorithm( int nAlgorithm);
    void setAttribute(int nAttribute);
    cv::Mat getResultMat() const;
private:
    VectorOfRect                    _vecSrchWindow;
    float                           _fErrorTol;
    int                             _nThreshold;
    int                             _nAlgorithm;
    int                             _nAttribute;
    cv::Mat                         _matResult;
    int fitRect(const std::string &imagePath);
};