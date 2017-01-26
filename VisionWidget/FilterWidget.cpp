#include "FilterWidget.h"
#include "visionwidget.h"
#include "VisionAPI.h"
#include <QMessageBox>

using namespace AOI::Vision;

FilterWidget::FilterWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);

    bool bSigmaVisible = ( ToInt32 ( PR_FILTER_TYPE::GAUSSIAN_FILTER ) == ui.cbFilterType->currentIndex() );
    ui.labelSigmaX->setVisible ( bSigmaVisible );
    ui.labelSigmaY->setVisible ( bSigmaVisible );
    ui.lineEditSigmaX->setVisible ( bSigmaVisible );
    ui.lineEditSigmaY->setVisible ( bSigmaVisible );

    bool bBilaterialFilter = ( ToInt32 ( PR_FILTER_TYPE::GAUSSIAN_FILTER ) == ui.cbFilterType->currentIndex() );
    ui.labelDiameter->setVisible ( bBilaterialFilter );
    ui.lineEditDiameter->setVisible ( bBilaterialFilter );
    ui.labelSigmaColor->setVisible ( bBilaterialFilter );
    ui.lineEditSigmaColor->setVisible ( bBilaterialFilter );
    ui.labelSigmaSpace->setVisible ( bBilaterialFilter );
    ui.lineEditSigmaSpace->setVisible ( bBilaterialFilter );
}

FilterWidget::~FilterWidget()
{
}

std::string FilterWidget::myName() const
{
    return "FilterWidget";
}

void FilterWidget::on_btnRun_clicked()
{
    if ( _checkImage() == false )
        return;

    PR_FILTER_CMD stCmd;
    stCmd.matInput = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    stCmd.rectROI  = _pVisionView->getSelectedWindow();
    stCmd.enType = static_cast<PR_FILTER_TYPE>(ui.cbFilterType->currentIndex());
    stCmd.szKernel.width  = ui.lineEditKernalSizeX->text().toInt();
    stCmd.szKernel.height = ui.lineEditKernalSizeY->text().toInt();
    stCmd.dSigmaX = ui.lineEditSigmaX->text().toDouble();
    stCmd.dSigmaY = ui.lineEditSigmaY->text().toDouble();
    stCmd.nDiameter = ui.lineEditDiameter->text().toInt();
    stCmd.dSigmaColor = ui.lineEditSigmaColor->text().toDouble();
    stCmd.dSigmaSpace = ui.lineEditSigmaSpace->text().toDouble();

    PR_FILTER_RPY stRpy;
    if ( VisionStatus::OK == PR_Filter(&stCmd, &stRpy) )    {
        _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, stRpy.matResult );
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
    bool bKernelSizeVisible = ( ToInt32 ( PR_FILTER_TYPE::BILATERIAL_FILTER ) != index );
    ui.labelKernelSize->setVisible ( bKernelSizeVisible );
    ui.lineEditKernalSizeX->setVisible ( bKernelSizeVisible );
    ui.lineEditKernalSizeY->setVisible ( bKernelSizeVisible );

    //using namespace AOI;
    bool bSigmaVisible = ( ToInt32 ( PR_FILTER_TYPE::GAUSSIAN_FILTER ) == index );
    ui.labelSigmaX->setVisible ( bSigmaVisible );
    ui.labelSigmaY->setVisible ( bSigmaVisible );
    ui.lineEditSigmaX->setVisible ( bSigmaVisible );
    ui.lineEditSigmaY->setVisible ( bSigmaVisible );

    bool bBilaterialFilter = ( ToInt32 ( PR_FILTER_TYPE::BILATERIAL_FILTER ) == index );
    ui.labelDiameter->setVisible ( bBilaterialFilter );
    ui.lineEditDiameter->setVisible ( bBilaterialFilter );
    ui.labelSigmaColor->setVisible ( bBilaterialFilter );
    ui.lineEditSigmaColor->setVisible ( bBilaterialFilter );
    ui.labelSigmaSpace->setVisible ( bBilaterialFilter );
    ui.lineEditSigmaSpace->setVisible ( bBilaterialFilter );
}