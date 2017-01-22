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
protected:
    VisionView*                     _pVisionView;
private:
    
};

#endif // BASEPROCESSWIDGET_H
