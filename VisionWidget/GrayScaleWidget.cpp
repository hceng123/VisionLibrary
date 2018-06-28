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

std::string GrayScaleWidget::myName() const
{
    return "GrayScaleWidget";
}

QSize GrayScaleWidget::sizeHint() const
{
    return QSize(520, 52);
}

cv::Mat GrayScaleWidget::_convertImage()
{
    cv::Mat mat = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    cv::Rect rectROI = _pVisionView->getSelectedWindow();

    if ( rectROI.x < 0 || rectROI.y < 0 ||
        ( rectROI.x + rectROI.width  ) > mat.cols ||
        ( rectROI.y + rectROI.height ) > mat.rows )    {
        return mat;
    }

    PR_COLOR_TO_GRAY_CMD stCmd;
    PR_COLOR_TO_GRAY_RPY stRpy;
    cv::Mat matROI(mat, rectROI);
    stCmd.matInputImg = matROI;
    stCmd.stRatio.fRatioR = ui.lineEditRRatio->text().toFloat();
    stCmd.stRatio.fRatioG = ui.lineEditGRatio->text().toFloat();
    stCmd.stRatio.fRatioB = ui.lineEditBRatio->text().toFloat();
    PR_ColorToGray(&stCmd, &stRpy);
    cv::Mat matGray = stRpy.matResultImg;
    cv::Mat matResultImg;
    cv::cvtColor(matGray, matResultImg, CV_GRAY2BGR);
    matResultImg.copyTo(matROI);
    return mat;
}

void GrayScaleWidget::on_btnConvert_clicked()
{
    if ( _checkImage() == false )
        return;

    _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, _convertImage() );
}