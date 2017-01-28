#pragma once

#include "ui_EdgeDetectWidget.h"
#include "BaseProcessWidget.h"

class EdgeDetectWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    EdgeDetectWidget(QWidget *parent = Q_NULLPTR);
    ~EdgeDetectWidget();
    virtual std::string myName() const override;

private slots:
    void on_btnRun_clicked();

private:
    Ui::EdgeDetectWidget ui;
};
