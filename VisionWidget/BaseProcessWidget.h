#ifndef BASEPROCESSWIDGET_H
#define BASEPROCESSWIDGET_H

#include <QWidget>
#include "VisionView.h"

class BaseProcessWidget : public QWidget
{
    Q_OBJECT

public:
    BaseProcessWidget(QWidget *parent);
    ~BaseProcessWidget();
    virtual void setVisionView(VisionView* pVisionView);
    virtual std::string myName() const = 0;
protected:
    bool                   _checkImage();    
    static std::string     _strCurrentWidgetName;
protected:
    VisionView*                     _pVisionView;
};

#endif // BASEPROCESSWIDGET_H
