#pragma once

#include "ui_ColorWidget.h"
#include "BaseProcessWidget.h"

class ColorWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    ColorWidget(QWidget *parent = Q_NULLPTR);
    ~ColorWidget();
    virtual std::string myName() const override;
private slots:
    void on_btnPickColor_clicked();
    void on_PickColor();
private:
    Ui::ColorWidget ui;
};
