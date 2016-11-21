#pragma once
#include "visionview.h"
class Procedure
{
public:
    Procedure(VisionView *pVisionView);
    virtual ~Procedure();
    virtual int run() = 0;
private:
    VisionView      *_pVisionView;
};

