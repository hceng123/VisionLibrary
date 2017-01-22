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

void ThresholdWidget::on_btnAutoThreshold_clicked()
{
    assert ( nullptr != _pVisionView );

    if ( _pVisionView->getMat().empty() )   {
        QMessageBox::information(this, "Vision Widget", "Please select an image first!", "Quit");
        return;
    }

    PR_AUTO_THRESHOLD_CMD stCmd;
    PR_AUTO_THRESHOLD_RPY stRpy;

    stCmd.matInput = _pVisionView->getMat();
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
    _pVisionView->setResultMat ( _runThreshold ( threshold ) );
}

void ThresholdWidget::on_lineEditThreshold_returnPressed()
{
    int threshold = ui.lineEditThreshold->text().toInt();
    ui.sliderThreshold->setValue(threshold);

    _pVisionView->setResultMat ( _runThreshold ( threshold ) );
}

cv::Mat ThresholdWidget::_runThreshold(int nThreshold)
{
    cv::Mat mat = _pVisionView->getMat().clone();
    if ( mat.empty() )
        return mat;

    cv::Mat matROI(mat, _pVisionView->getSelectedWindow());
    bool bReverseThreshold = ui.checkBoxReverseThres->isChecked();

    cv::Mat matThreshold;
    cv::threshold ( matROI, matThreshold, nThreshold, PR_MAX_GRAY_LEVEL, bReverseThreshold ? THRESH_BINARY_INV : THRESH_BINARY );
    
    matThreshold.copyTo ( matROI  );
    return mat;
}

void ThresholdWidget::on_btnApply_clicked()
{
    int threshold = ui.lineEditThreshold->text().toInt();
    ui.sliderThreshold->setValue(threshold);

    _pVisionView->setMat ( _runThreshold ( threshold ) );
}