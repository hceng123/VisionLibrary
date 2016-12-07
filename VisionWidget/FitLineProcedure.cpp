#include "FitLineProcedure.h"
#include "messageboxdialog.h"
#include "constants.h"
#include <QMessageBox>

using namespace AOI::Vision;

FitLineProcedure::FitLineProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

FitLineProcedure::~FitLineProcedure()
{
}

int FitLineProcedure::run(const std::string &imagePath)
{
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setWindowTitle("Fit Line");
	pMessageBox->SetMessageText1("Please input the search window");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_RECT_SRCH_WINDOW);
    _pVisionView->setCurrentSrchWindowIndex(0);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }	
   
    VectorOfRect vecResult = _pVisionView->getVecSrchWindow();
    if ( vecResult.size() <= 0 )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }
    _rectSrchWindow = vecResult[0];

    int nStatus = fitLine(imagePath);

    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
    return nStatus;
}

int FitLineProcedure::fitLine(const std::string &imagePath)
{
    PR_FIT_LINE_CMD stCmd;
	stCmd.matInput = cv::imread(imagePath);
	stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
	stCmd.fErrTol = _fErrorTol;
	stCmd.rectROI = _rectSrchWindow;
	stCmd.nThreshold = _nThreshold;

	PR_FIT_LINE_RPY stRpy;
	VisionStatus visionStatus = PR_FitLine(&stCmd, &stRpy);
	if (VisionStatus::OK != visionStatus)	{
		std::cout << "Failed to fit circle, VisionStatus = " << stRpy.nStatus << std::endl;
		return static_cast<int> ( visionStatus );
	}
	_matResult = stRpy.matResult;	
    return ToInt(STATUS::OK);
}

void FitLineProcedure::setErrTol(float fErrTol)
{
    _fErrorTol = fErrTol;
}

void FitLineProcedure::setThreshold(int nThreshold)
{
    _nThreshold = nThreshold;
}

void FitLineProcedure::setAlgorithm(int nAlgorithm)
{
    _nAlgorithm = nAlgorithm;
}

cv::Mat FitLineProcedure::getResultMat() const
{
    return _matResult;
}