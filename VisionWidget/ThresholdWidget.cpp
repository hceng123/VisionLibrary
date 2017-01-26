#include "ThresholdWidget.h"
#include <QMessageBox>

ThresholdWidget::ThresholdWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);

    _ptrThresValidator  = std::make_unique<QIntValidator>(1, PR_MAX_GRAY_LEVEL);
    ui.lineEditThreshold->setValidator ( _ptrThresValidator.get() );
}

ThresholdWidget::~ThresholdWidget()
{
}

std::string ThresholdWidget::myName() const
{
    return "ThresholdWidget";
}

void ThresholdWidget::on_btnAutoThreshold_clicked()
{
    if ( _checkImage() == false )
        return;

    PR_AUTO_THRESHOLD_CMD stCmd;
    PR_AUTO_THRESHOLD_RPY stRpy;

    stCmd.matInput = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    stCmd.rectROI = _pVisionView->getSelectedWindow();
    stCmd.nThresholdNum = ui.cbAutoThresLevel->currentText().toInt();

    PR_AutoThreshold(&stCmd, &stRpy);
    std::string strThreshold;
    for ( auto threshold : stRpy.vecThreshold )
        strThreshold += std::to_string(threshold) + " ";
    ui.lineEditAutoThresholdResult->setText(strThreshold.c_str());
}

void ThresholdWidget::on_sliderThreshold_valueChanged(int threshold)
{
    ui.lineEditThreshold->setText(std::to_string(threshold).c_str());
    _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, _runThreshold ( threshold ) );
}

void ThresholdWidget::on_lineEditThreshold_returnPressed()
{
    int threshold = ui.lineEditThreshold->text().toInt();
    ui.sliderThreshold->setValue(threshold);

    _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, _runThreshold ( threshold ) );
}

cv::Mat ThresholdWidget::_runThreshold(int nThreshold)
{
    cv::Mat mat = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL).clone();
    if ( mat.empty() )
        return mat;

    if ( _checkImage() == false )
        return mat;

    cv::Mat matROI(mat, _pVisionView->getSelectedWindow());
    bool bReverseThreshold = ui.checkBoxReverseThres->isChecked();

    cv::Mat matThreshold;
    cv::threshold ( matROI, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, bReverseThreshold ? THRESH_BINARY_INV : THRESH_BINARY );
    
    matThreshold.copyTo ( matROI  );
    return mat;
}

void ThresholdWidget::on_checkBoxReverseThres_clicked()
{
    int threshold = ui.lineEditThreshold->text().toInt();
    _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, _runThreshold ( threshold ) );
}