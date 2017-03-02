#pragma once

#include "ui_ColorWidget.h"
#include "BaseProcessWidget.h"

class ColorWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    ColorWidget(QWidget *parent = Q_NULLPTR);
    ~ColorWidget();

private:
    Ui::ColorWidget ui;
};
