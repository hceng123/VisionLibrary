#pragma once
#include "Procedure.h"
class FitCircleProcedure :
    public Procedure
{
public:
    FitCircleProcedure(VisionView *pVisionView);
    ~FitCircleProcedure();
    virtual int run() override;
};

