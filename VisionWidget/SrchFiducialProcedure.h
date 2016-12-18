#pragma once
#include "Procedure.h"

class SrchFiducialProcedure :
    public Procedure
{
public:
    SrchFiducialProcedure(VisionView *pVisionView);
    ~SrchFiducialProcedure();
    virtual int run(const std::string &imagePath) override;
    cv::Mat getResultMat() const;
    void setFiducialSize(float fFiducialSize);
    void setFiducialMarge(float fMargin);
    void setFicucialType(int           nFiducialType);
private:
    cv::Rect                        _rectSrchWindow;
    float                           _fFiducialSize;
    float                           _fFiducialMargin;
    int                             _nFiducialType;
    cv::Mat                         _matResult;
    int srchFiducial(const std::string &imagePath);
};