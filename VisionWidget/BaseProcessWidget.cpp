#include "BaseProcessWidget.h"
#include <QMessageBox>

std::string BaseProcessWidget::_strCurrentWidgetName = "undefined";

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

bool BaseProcessWidget::_checkImage()
{
    assert ( nullptr != _pVisionView );

    if ( _pVisionView->getMat().empty() )
    {
        QMessageBox::information(nullptr, "Vision Widget", "Please select an image first!", "Quit");
        return false;
    }

    if ( _strCurrentWidgetName != myName() )    {
        _pVisionView->applyIntermediateResult();
        _strCurrentWidgetName = myName();
    }
    return true;
}