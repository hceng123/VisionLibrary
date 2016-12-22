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
    cv::Mat getResultMat() const;
private:
    VectorOfRect                    _vecSrchWindow;
    float                           _fErrorTol;
    int                             _nThreshold;
    int                             _nAlgorithm;
    cv::Mat                         _matResult;
    int fitRect(const std::string &imagePath);
};