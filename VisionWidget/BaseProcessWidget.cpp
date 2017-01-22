#include "BaseProcessWidget.h"

BaseProcessWidget::BaseProcessWidget(QWidget *parent)
    : QWidget(parent)
{
}

BaseProcessWidget::~BaseProcessWidget()
{
}

void BaseProcessWidget::setVisionView(VisionView* pVisionView)
{
    _pVisionView = pVisionView;
}