#include "CCWidget.h"
#include <QMessageBox>

CCWidget::CCWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);
}

CCWidget::~CCWidget()
{
}

std::string CCWidget::myName() const
{
    return "CCWidget";
}

VisionStatus CCWidget::_runRemoveCC(cv::Mat &mat)
{
    PR_REMOVE_CC_CMD stCmd;
    PR_REMOVE_CC_RPY stRpy;
    stCmd.matInputImg = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    stCmd.rectROI  = _pVisionView->getSelectedWindow();
    stCmd.fAreaThreshold = ui.lineEditAreaThreshold->text().toFloat();
    stCmd.nConnectivity = ui.cbConnectivity->currentText().toInt();    
    
    if ( VisionStatus::OK == PR_RemoveCC( &stCmd, &stRpy ) )    {
        mat = stRpy.matResultImg;
    }else  {
        PR_GET_ERROR_INFO_RPY stErrStrRpy;
        PR_GetErrorInfo( stRpy.enStatus, &stErrStrRpy );
        QMessageBox::critical(this, "Filter failed", stErrStrRpy.achErrorStr, "Quit");
    }
    return stRpy.enStatus;
}

void CCWidget::on_btnRun_clicked()
{
    if ( _checkImage() == false )
        return;

    cv::Mat mat;
    VisionStatus enStatus = _runRemoveCC ( mat );
    if ( VisionStatus::OK == enStatus )
        _pVisionView->setMat ( VisionView::DISPLAY_SOURCE::INTERMEDIATE, mat );
}