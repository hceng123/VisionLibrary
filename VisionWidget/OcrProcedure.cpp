#include "OcrProcedure.h"
#include "messageboxdialog.h"
#include "constants.h"
#include <QMessageBox>

using namespace AOI::Vision;

OcrProcedure::OcrProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

OcrProcedure::~OcrProcedure()
{
}

int OcrProcedure::run(const std::string &imagePath)
{
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setWindowTitle("Ocr");
	pMessageBox->SetMessageText1("Please input the search window");
    pMessageBox->SetMessageText2("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_MULTIPLE_WINDOW);
    _pVisionView->setCurrentSrchWindowIndex(0);
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return ToInt(STATUS::NOK);
    }	
   
    VectorOfRect vecResult = _pVisionView->getVecSrchWindow();
    if ( vecResult.size() <= 0 )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
        return ToInt(STATUS::NOK);
    }
    _rectSrchWindow = vecResult[0];

    int nStatus = ocr(imagePath);

    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
    return nStatus;
}

int OcrProcedure::ocr(const std::string &imagePath)
{
    PR_OCR_CMD stCmd;
	stCmd.matInputImg = _pVisionView->getMat();
    stCmd.enDirection = static_cast<PR_DIRECTION>(_nTextDirection);
	stCmd.rectROI = _rectSrchWindow;

	PR_OCR_RPY stRpy;
	VisionStatus visionStatus = PR_Ocr(&stCmd, &stRpy);
	if (VisionStatus::OK != visionStatus)	{
		std::cout << "Failed to do OCR, VisionStatus = " << ToInt32( stRpy.enStatus ) << std::endl;
		return static_cast<int> ( visionStatus );
	}
    _strResult = stRpy.strResult;
    return ToInt(STATUS::OK);
}

std::string OcrProcedure::getResult() const
{
    return _strResult;
}

void OcrProcedure::setTextDirection(int nTextDirection)
{
    _nTextDirection = nTextDirection;
}