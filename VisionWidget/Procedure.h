#pragma once
#include "visionview.h"
#include "VisionAPI.h"

class Procedure
{
public:
    Procedure(VisionView *pVisionView);
    virtual ~Procedure();
    virtual int run(const std::string &imagePath) = 0;
protected:
    VisionView      *_pVisionView;
};
