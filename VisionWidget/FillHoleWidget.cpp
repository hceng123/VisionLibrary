#include "FillHoleWidget.h"
#include <QMessageBox>

FillHoleWidget::FillHoleWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);
}

FillHoleWidget::~FillHoleWidget()
{
}

std::string FillHoleWidget::myName() const
{
    return "FillHoleWidget";
}

void FillHoleWidget::on_btnRun_clicked()
{
    if ( _checkImage() == false )
        return;

    PR_FILL_HOLE_CMD stCmd;
    stCmd.matInput = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    stCmd.rectROI  = _pVisionView->getSelectedWindow();
    stCmd.enMethod = static_cast<PR_FILL_HOLE_METHOD>(ui.cbAlgorithm->currentIndex());
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ui.cbObjectAttribute->currentIndex());
    //stCmd.szKernel.width  = ui.lineEditKernalSizeX->text().toInt();
    //stCmd.szKernel.height = ui.lineEditKernalSizeY->text().toInt();
    //stCmd.dSigmaX = ui.lineEditSigmaX->text().toDouble();
    //stCmd.dSigmaY = ui.lineEditSigmaY->text().toDouble();
    //stCmd.nDiameter = ui.lineEditDiameter->text().toInt();
    //stCmd.dSigmaColor = ui.lineEditSigmaColor->text().toDouble();
    //stCmd.dSigmaSpace = ui.lineEditSigmaSpace->text().toDouble();

    PR_FILL_HOLE_RPY stRpy;
    if ( VisionStatus::OK == PR_FillHole(&stCmd, &stRpy) )    {
        _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, stRpy.matResult );
    }
    else
    {
        PR_GET_ERROR_STR_RPY stErrStrRpy;
        PR_GetErrorStr( stRpy.enStatus, &stErrStrRpy );
        QMessageBox::critical(nullptr, "Filter failed", stErrStrRpy.achErrorStr, "Quit");
    }
}

void FillHoleWidget::on_cbAlgorithm_currentIndexChanged(int index)
{
    //bool bKernelSizeVisible = ( ToInt32 ( PR_FILTER_TYPE::BILATERIAL_FILTER ) != index );
    //ui.labelKernelSize->setVisible ( bKernelSizeVisible );
    //ui.lineEditKernalSizeX->setVisible ( bKernelSizeVisible );
    //ui.lineEditKernalSizeY->setVisible ( bKernelSizeVisible );

    ////using namespace AOI;
    //bool bSigmaVisible = ( ToInt32 ( PR_FILTER_TYPE::GAUSSIAN_FILTER ) == index );
    //ui.labelSigmaX->setVisible ( bSigmaVisible );
    //ui.labelSigmaY->setVisible ( bSigmaVisible );
    //ui.lineEditSigmaX->setVisible ( bSigmaVisible );
    //ui.lineEditSigmaY->setVisible ( bSigmaVisible );

    //bool bBilaterialFilter = ( ToInt32 ( PR_FILTER_TYPE::BILATERIAL_FILTER ) == index );
    //ui.labelDiameter->setVisible ( bBilaterialFilter );
    //ui.lineEditDiameter->setVisible ( bBilaterialFilter );
    //ui.labelSigmaColor->setVisible ( bBilaterialFilter );
    //ui.lineEditSigmaColor->setVisible ( bBilaterialFilter );
    //ui.labelSigmaSpace->setVisible ( bBilaterialFilter );
    //ui.lineEditSigmaSpace->setVisible ( bBilaterialFilter );
}
