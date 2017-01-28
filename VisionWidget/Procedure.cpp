#include "Procedure.h"


Procedure::Procedure(VisionView *pVisionView):_pVisionView(pVisionView)
{
    _pVisionView->setState ( VisionView::VISION_VIEW_STATE::TEST_VISION_LIBRARY );
    _pVisionView->applyIntermediateResult();
}

Procedure::~Procedure()
{
}
