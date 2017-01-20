#include "FilterWidget.h"
#include "visionwidget.h"
#include "VisionAPI.h"
#include <QMessageBox>

using namespace AOI::Vision;

FilterWidget::FilterWidget(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);

    bool bSigmaVisible = ( ToInt32 ( PR_FILTER_TYPE::GAUSSIAN_FILTER ) == ui.cbFilterType->currentIndex() );
    ui.labelSigmaX->setVisible ( bSigmaVisible );
    ui.labelSigmaY->setVisible ( bSigmaVisible );
    ui.lineEditSigmaX->setVisible ( bSigmaVisible );
    ui.lineEditSigmaY->setVisible ( bSigmaVisible );
}

FilterWidget::~FilterWidget()
{
}

void FilterWidget::setVisionView(VisionView* pVisionView)
{
    _pVisionView = pVisionView;
}

void FilterWidget::on_btnRun_clicked()
{
    //VisionWidget *pVW = qobject_cast<VisionWidget *>(this->parent());
    VisionWidget *pVW = (VisionWidget *)(parent());
    if ( nullptr == _pVisionView )
        return;

    if ( _pVisionView->getMat().empty() )   {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    PR_FILTER_CMD stCmd;
    stCmd.matInput = _pVisionView->getMat();
    stCmd.rectROI  = _pVisionView->getSelectedWindow();
    stCmd.enType = static_cast<PR_FILTER_TYPE>(ui.cbFilterType->currentIndex());
    stCmd.szKernel.width  = ui.lineEditKernalSizeX->text().toInt();
    stCmd.szKernel.height = ui.lineEditKernalSizeY->text().toInt();
    stCmd.dSigmaX = ui.lineEditSigmaX->text().toDouble();
    stCmd.dSigmaY = ui.lineEditSigmaY->text().toDouble();

    PR_FILTER_RPY stRpy;
    if ( VisionStatus::OK == PR_Filter(&stCmd, &stRpy) )    {
        _pVisionView->setMat ( stRpy.matResult );
    }
    else
    {
        PR_GET_ERROR_STR_RPY stErrStrRpy;
        PR_GetErrorStr( stRpy.enStatus, &stErrStrRpy );
        QMessageBox::critical(nullptr, "Filter failed", stErrStrRpy.achErrorStr, "Quit");
    }
}

void FilterWidget::on_cbFilterType_currentIndexChanged(int index)
{
    //using namespace AOI;
    bool bSigmaVisible = ( ToInt32 ( PR_FILTER_TYPE::GAUSSIAN_FILTER ) == index );
    ui.labelSigmaX->setVisible ( bSigmaVisible );
    ui.labelSigmaY->setVisible ( bSigmaVisible );
    ui.lineEditSigmaX->setVisible ( bSigmaVisible );
    ui.lineEditSigmaY->setVisible ( bSigmaVisible );
}