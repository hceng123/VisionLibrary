#include "FilterWidget.h"
#include "visionwidget.h"
#include "VisionAPI.h";

using namespace AOI::Vision;

FilterWidget::FilterWidget(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);
}

FilterWidget::~FilterWidget()
{
}

void FilterWidget::on_btnRun_clicked()
{
    VisionWidget *pVW = (VisionWidget *)this->parentWidget();
    PR_FILTER_CMD stCmd;
    stCmd.szKernel.width = ui.lineEditKernalSizeX->text().toInt();
}