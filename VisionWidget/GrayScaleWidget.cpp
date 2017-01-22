#include "GrayScaleWidget.h"
#include <QMessageBox>

GrayScaleWidget::GrayScaleWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);
}

GrayScaleWidget::~GrayScaleWidget()
{
}


cv::Mat GrayScaleWidget::_convertImage()
{
    assert ( nullptr != _pVisionView );

    cv::Mat mat = _pVisionView->getMat();
    if (mat.empty())
    {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return mat;
    }

    PR_COLOR_TO_GRAY_CMD stCmd;
    PR_COLOR_TO_GRAY_RPY stRpy;
    stCmd.matInput = mat;
    stCmd.stRatio.fRatioR = ui.lineEditRRatio->text().toFloat();
    stCmd.stRatio.fRatioG = ui.lineEditGRatio->text().toFloat();
    stCmd.stRatio.fRatioB = ui.lineEditBRatio->text().toFloat();
    PR_ColorToGray(&stCmd, &stRpy);
    cv::Mat matGray = stRpy.matResult;
    return matGray;
}

void GrayScaleWidget::on_btnConvert_clicked()
{
    _pVisionView->setResultMat ( _convertImage() );
}

void GrayScaleWidget::on_btnApply_clicked()
{
    _pVisionView->setMat ( _convertImage() );
}