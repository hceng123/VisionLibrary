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

std::string GrayScaleWidget::MyName() const
{
    return "GrayScaleWidget";
}

QSize GrayScaleWidget::sizeHint() const
{
    return QSize(520, 52);
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

    cv::Rect rectROI = _pVisionView->getSelectedWindow();

    if ( rectROI.x < 0 || rectROI.y < 0 ||
        ( rectROI.x + rectROI.width  ) > mat.cols ||
        ( rectROI.y + rectROI.height ) > mat.rows )    {
        return mat;
    }

    PR_COLOR_TO_GRAY_CMD stCmd;
    PR_COLOR_TO_GRAY_RPY stRpy;
    cv::Mat matROI(mat, rectROI );
    stCmd.matInput = matROI;
    stCmd.stRatio.fRatioR = ui.lineEditRRatio->text().toFloat();
    stCmd.stRatio.fRatioG = ui.lineEditGRatio->text().toFloat();
    stCmd.stRatio.fRatioB = ui.lineEditBRatio->text().toFloat();
    PR_ColorToGray(&stCmd, &stRpy);
    cv::Mat matGray = stRpy.matResult;
    cv::Mat matResult;
    cv::cvtColor(matGray, matResult, CV_GRAY2BGR);
    matResult.copyTo(matROI);
    return mat;
}

void GrayScaleWidget::on_btnConvert_clicked()
{
    _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, _convertImage() );
}

void GrayScaleWidget::on_btnApply_clicked()
{
    //_pVisionView->setMat ( _convertImage() );
}