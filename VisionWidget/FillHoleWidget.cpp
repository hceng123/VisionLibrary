#include "FillHoleWidget.h"
#include <QMessageBox>

FillHoleWidget::FillHoleWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);
    on_cbAlgorithm_currentIndexChanged(0);
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
    stCmd.matInputImg = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    stCmd.rectROI  = _pVisionView->getSelectedWindow();
    stCmd.enMethod = static_cast<PR_FILL_HOLE_METHOD>(ui.cbAlgorithm->currentIndex());
    stCmd.enAttribute = static_cast<PR_OBJECT_ATTRIBUTE>(ui.cbObjectAttribute->currentIndex());
    stCmd.enMorphShape = static_cast<cv::MorphShapes>(ui.cbMorphShape->currentIndex());
    int nSize = ui.lineEditKernalSize->text().toInt();
    stCmd.szMorphKernel  = cv::Size( nSize, nSize );
    stCmd.nMorphIteration = ui.lineEditIteration->text().toInt();

    PR_FILL_HOLE_RPY stRpy;
    if ( VisionStatus::OK == PR_FillHole(&stCmd, &stRpy) )    {
        _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, stRpy.matResultImg );
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
    bool bKernelSizeVisible = ( ToInt32 ( PR_FILL_HOLE_METHOD::MORPHOLOGY ) == index );
    ui.labelMorphShape->setVisible ( bKernelSizeVisible );
    ui.cbMorphShape->setVisible ( bKernelSizeVisible );
    ui.labelKernelSize->setVisible ( bKernelSizeVisible );
    ui.lineEditKernalSize->setVisible ( bKernelSizeVisible );
    ui.labelIteration->setVisible ( bKernelSizeVisible );
    ui.lineEditIteration->setVisible ( bKernelSizeVisible );
}
