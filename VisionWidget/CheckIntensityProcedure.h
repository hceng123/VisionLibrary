#pragma once
#include "Procedure.h"

class CheckIntensityProcedure :
    public Procedure
{
public:
    CheckIntensityProcedure(VisionView *pVisionView);
    ~CheckIntensityProcedure();
    virtual int run(const std::string &imagePath) override { return 0; };
    int run(const cv::Mat &mat);
private:
    int checkIntensity(const cv::Mat &mat);
    PR_Line   _lineOfIntensityCheck;
};