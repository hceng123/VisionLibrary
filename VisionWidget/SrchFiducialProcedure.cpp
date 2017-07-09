#include "SrchFiducialProcedure.h"
#include "messageboxdialog.h"
#include "constants.h"
#include <QMessageBox>
#include <QApplication>

using namespace AOI::Vision;

SrchFiducialProcedure::SrchFiducialProcedure(VisionView *pVisionView) : Procedure(pVisionView)
{
}

SrchFiducialProcedure::~SrchFiducialProcedure()
{
}

int SrchFiducialProcedure::run(const std::string &imagePath)
{
    QRect rect = qApp->activeWindow()->geometry();
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(rect.x() + POS_X, rect.y() + POS_Y, pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Search Fiducial");
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

    int nStatus = srchFiducial(imagePath);

    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
    return nStatus;
}

int SrchFiducialProcedure::srchFiducial(const std::string &imagePath)
{
    PR_SRCH_FIDUCIAL_MARK_CMD stCmd;
	stCmd.matInputImg = _pVisionView->getMat();
    stCmd.rectSrchRange = _rectSrchWindow;
	stCmd.enType = static_cast<PR_FIDUCIAL_MARK_TYPE>(_nFiducialType);
	stCmd.fSize = _fFiducialSize;	
	stCmd.fMargin = _fFiducialMargin;

	PR_SRCH_FIDUCIAL_MARK_RPY stRpy;
	VisionStatus visionStatus = PR_SrchFiducialMark(&stCmd, &stRpy);
	if (VisionStatus::OK != visionStatus)	{
		std::cout << "Failed to search fiducial mark, VisionStatus = " << ToInt32(stRpy.enStatus) << std::endl;
		return static_cast<int> ( visionStatus );
	}
	_matResult = stRpy.matResultImg;	
    return ToInt(STATUS::OK);
}

void SrchFiducialProcedure::setFiducialSize(float fFiducialSize)
{
    _fFiducialSize = fFiducialSize;
}

void SrchFiducialProcedure::setFiducialMarge(float fMargin)
{
    _fFiducialMargin = fMargin;
}

void SrchFiducialProcedure::setFicucialType(int           nFiducialType)
{
    _nFiducialType = nFiducialType;
}

cv::Mat SrchFiducialProcedure::getResultMat() const
{
    return _matResult;
}