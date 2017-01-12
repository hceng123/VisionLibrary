#pragma once
#include "Procedure.h"

class FitLineProcedure :
    public Procedure
{
public:
    FitLineProcedure(VisionView *pVisionView);
    ~FitLineProcedure();
    virtual int run(const std::string &imagePath) override;
    void setErrTol(float fErrTol);
    void setThreshold(int nThreshold);
    void setAlgorithm( int nAlgorithm);
    void setAttribute(int nAttribute);
    cv::Mat getResultMat() const;
private:
    cv::Rect                        _rectSrchWindow;
    float                           _fErrorTol;
    int                             _nThreshold;
    int                             _nAlgorithm;
    int                             _nAttribute;
    cv::Mat                         _matResult;
    int fitLine(const std::string &imagePath);
};