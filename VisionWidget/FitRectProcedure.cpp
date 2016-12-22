#include "FitRectProcedure.h"
#include "messageboxdialog.h"
#include "constants.h"
#include <QMessageBox>
#include <QApplication>

using namespace AOI::Vision;

FitRectProcedure::FitRectProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

FitRectProcedure::~FitRectProcedure()
{
}

int FitRectProcedure::run(const std::string &imagePath)
{
    QRect rect = qApp->activeWindow()->geometry();
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(rect.x() + POS_X, rect.y() + POS_Y, pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Fit Parallel Line");
	pMessageBox->SetMessageText1("Please input the line 1 search window");
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

    pMessageBox->SetMessageText1("Please input the line 2 search window");
    pMessageBox->SetMessageText2("It should be parallel to the last line");
    pMessageBox->SetMessageText3("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_RECT_SRCH_WINDOW);
    _pVisionView->setCurrentSrchWindowIndex(1);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }

    pMessageBox->SetMessageText1("Please input the line 3 search window");
    pMessageBox->SetMessageText2("It should be perpendicular to the last line");
    pMessageBox->SetMessageText3("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_RECT_SRCH_WINDOW);
    _pVisionView->setCurrentSrchWindowIndex(2);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }

    pMessageBox->SetMessageText1("Please input the line 4 search window");
    pMessageBox->SetMessageText2("It should be parallel to the last line");
    pMessageBox->SetMessageText3("Press and drag the left mouse buttont to input");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::SET_RECT_SRCH_WINDOW);
    _pVisionView->setCurrentSrchWindowIndex(3);

	iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }
   
    _vecSrchWindow = _pVisionView->getVecSrchWindow();
    if ( _vecSrchWindow.size() < PR_RECT_EDGE_COUNT )   {
        pMessageBox->SetMessageText1("The input is invalid");
        pMessageBox->SetMessageText2("");
        pMessageBox->exec();
        _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
        return ToInt(STATUS::NOK);
    }

    int nStatus = fitRect(imagePath);

    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::UNDEFINED);
    return nStatus;
}

int FitRectProcedure::fitRect(const std::string &imagePath)
{
    PR_FIT_RECT_CMD stCmd;
	stCmd.matInput = cv::imread(imagePath);
	stCmd.enRmNoiseMethod = PR_RM_FIT_NOISE_METHOD::ABSOLUTE_ERR;
	stCmd.fErrTol = _fErrorTol;
	stCmd.rectArrROI[0] = _vecSrchWindow[0];
    stCmd.rectArrROI[1] = _vecSrchWindow[1];
    stCmd.rectArrROI[2] = _vecSrchWindow[2];
    stCmd.rectArrROI[3] = _vecSrchWindow[3];
	stCmd.nThreshold = _nThreshold;

	PR_FIT_RECT_RPY stRpy;
	VisionStatus visionStatus = PR_FitRect(&stCmd, &stRpy);
	if (VisionStatus::OK != visionStatus)	{
		std::cout << "Failed to fit parallel line, VisionStatus = " << stRpy.nStatus << std::endl;
		return static_cast<int> ( visionStatus );
	}
	_matResult = stRpy.matResult;	
    return ToInt(STATUS::OK);
}

void FitRectProcedure::setErrTol(float fErrTol)
{
    _fErrorTol = fErrTol;
}

void FitRectProcedure::setThreshold(int nThreshold)
{
    _nThreshold = nThreshold;
}

void FitRectProcedure::setAlgorithm(int nAlgorithm)
{
    _nAlgorithm = nAlgorithm;
}

cv::Mat FitRectProcedure::getResultMat() const
{
    return _matResult;
}