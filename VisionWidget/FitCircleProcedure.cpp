#include "FitCircleProcedure.h"
#include "messageboxdialog.h"
#include "constants.h"
#include <QMessageBox>

using namespace AOI::Vision;

FitCircleProcedure::FitCircleProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

FitCircleProcedure::~FitCircleProcedure()
{
}

int FitCircleProcedure::run(const std::string &imagePath)
{
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setWindowTitle("Fit circle");
	pMessageBox->SetMessageText1("Please input the center of fit range");
    pMessageBox->SetMessageText2("Click left mouse button on vision view to select!");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_CIRCLE_CTR);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }
	
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_CIRCLE_INNER_RADIUS);
    pMessageBox->SetMessageText1("Please input the inner radius of fit range");
    pMessageBox->SetMessageText2("Click left mouse button on vision view to select!");
    pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }

    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_CIRCLE_OUTTER_RADIUS);
    pMessageBox->SetMessageText1("Please input the outter radius of fit range");
    pMessageBox->SetMessageText2("Click left mouse button on vision view to select!");
    pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }
    _pVisionView->getFitCircleRange ( _ptCircleCtr, _fInnerRangeRadius, _fOutterRangeRadius );
    if ( _ptCircleCtr.x <= 0 || _ptCircleCtr.y <= 0 || _fInnerRangeRadius <= 2 || _fOutterRangeRadius <= _fInnerRangeRadius )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }
    int nStatus = fitCircle(imagePath);
    return nStatus;
}

int FitCircleProcedure::fitCircle(const std::string &imagePath)
{
    PR_FIT_CIRCLE_CMD stCmd;
	stCmd.matInput = cv::imread(imagePath);
	stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
	stCmd.fErrTol = _fErrorTol;
	stCmd.ptRangeCtr = _ptCircleCtr;
	stCmd.fRangeInnterRadius = _fInnerRangeRadius;
	stCmd.fRangeOutterRadius = _fOutterRangeRadius;
	stCmd.bAutothreshold = false;
	stCmd.nThreshold = _nThreshold;
	stCmd.enMethod = PR_FIT_CIRCLE_METHOD(_nAlgorithm);
	stCmd.nMaxRansacTime = 20;

	PR_FIT_CIRCLE_RPY stRpy;
	VisionStatus visionStatus = PR_FitCircle(&stCmd, &stRpy);
	if (VisionStatus::OK != visionStatus)	{
		std::cout << "Failed to fit circle, VisionStatus = " << stRpy.nStatus << std::endl;
		return static_cast<int> ( visionStatus );
	}
	_matResult = stRpy.matResult;	
    return ToInt(STATUS::OK);
}

void FitCircleProcedure::setErrTol(float fErrTol)
{
    _fErrorTol = fErrTol;
}

void FitCircleProcedure::setThreshold(int nThreshold)
{
    _nThreshold = nThreshold;
}

void FitCircleProcedure::setAlgorithm(int nAlgorithm)
{
    _nAlgorithm = nAlgorithm;
}

cv::Mat FitCircleProcedure::getResultMat() const
{
    return _matResult;
}