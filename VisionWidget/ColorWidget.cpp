#include "ColorWidget.h"
#include "messageboxdialog.h"
#include <QMessageBox>

ColorWidget::ColorWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);
}

ColorWidget::~ColorWidget()
{
}

std::string ColorWidget::myName() const
{
    return "ColorWidget";
}

void ColorWidget::on_btnPickColor_clicked()
{
    QRect rect = this->geometry();
    QPoint pt = this->mapToGlobal(QPoint(10, 10));
	std::unique_ptr<MessageBoxDialog> pMessageBox = std::make_unique<MessageBoxDialog>();
    pMessageBox->setGeometry(pt.x(), pt.y(), pMessageBox->size().width(), pMessageBox->size().height());
    pMessageBox->setWindowTitle("Pick color");
	pMessageBox->SetMessageText1("Please click left mouse button to pick color");
	pMessageBox->setWindowFlags(Qt::WindowStaysOnTopHint);
	pMessageBox->show();
	pMessageBox->raise();
	pMessageBox->activateWindow();
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::PICK_COLOR);
    connect(_pVisionView, SIGNAL(PickColor()), this, SLOT ( on_PickColor() ) );
	int iReturn = pMessageBox->exec();
    if ( iReturn != QDialog::Accepted ) {
        cv::Mat mat = _pVisionView->getMat();
        _pVisionView->setMat(VisionView::DISPLAY_SOURCE::ORIGINAL, mat );
    }
    _pVisionView->setTestVisionState(VisionView::TEST_VISION_STATE::IDLE);
}

void ColorWidget::on_PickColor()
{
    PR_PICK_COLOR_CMD stCmd;
    stCmd.matInput = _pVisionView->getMat(VisionView::DISPLAY_SOURCE::ORIGINAL);
    stCmd.rectROI  = _pVisionView->getSelectedWindow();
    stCmd.ptPick = _pVisionView->getClickedPoint();
    stCmd.nColorDiff = ui.lineEditColorDiff->text().toInt();
    stCmd.nGrayDiff = ui.lineEditGrayDiff->text().toInt();

    PR_PICK_COLOR_RPY stRpy;
    if ( VisionStatus::OK == PR_PickColor(&stCmd, &stRpy) )    {
        _pVisionView->setMat(VisionView::DISPLAY_SOURCE::INTERMEDIATE, stRpy.matResult );
    }
    else
    {
        PR_GET_ERROR_STR_RPY stErrStrRpy;
        PR_GetErrorStr( stRpy.enStatus, &stErrStrRpy );
        QMessageBox::critical(nullptr, "Pick color failed", stErrStrRpy.achErrorStr, "Quit");
    }
}