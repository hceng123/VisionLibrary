#include "EdgeDetectWidget.h"
#include <QMessageBox>

EdgeDetectWidget::EdgeDetectWidget(QWidget *parent)
    : BaseProcessWidget(parent)
{
    ui.setupUi(this);
}

EdgeDetectWidget::~EdgeDetectWidget()
{
}

std::string EdgeDetectWidget::myName() const
{
    return "EdgeDetectWidget";
}

void EdgeDetectWidget::on_btnRun_clicked()
{
    if ( !_checkImage() )
        return;

    PR_DETECT_EDGE_CMD stCmd;
    stCmd.matInput = _pVisionView->getMat();
    stCmd.rectROI = _pVisionView->getSelectedWindow();
    stCmd.nThreshold1 = ui.lineEditThreshold1->text().toInt();
    stCmd.nThreshold2 = ui.lineEditThreshold2->text().toInt();
    stCmd.nApertureSize = ui.lineEditApertureSize->text().toInt();

    PR_DETECT_EDGE_RPY stRpy;
    PR_DetectEdge ( &stCmd, &stRpy );

     if ( VisionStatus::OK == stRpy.enStatus )    {
        _pVisionView->setMat ( VisionView::DISPLAY_SOURCE::INTERMEDIATE, stRpy.matResult );
    }else  {
        PR_GET_ERROR_STR_RPY stErrStrRpy;
        PR_GetErrorStr( stRpy.enStatus, &stErrStrRpy );
        QMessageBox::critical(this, "Filter failed", stErrStrRpy.achErrorStr, "Quit");
    }
}