#include "BaseProcessWidget.h"
#include <QMessageBox>

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

bool BaseProcessWidget::_checkImage() const
{
    assert ( nullptr != _pVisionView );

    if ( _pVisionView->getMat().empty() )
    {
        QMessageBox::information(nullptr, "Vision Widget", "Please select an image first!", "Quit");
        return false;
    }
    return true;
}