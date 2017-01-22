#pragma once

#include <QWidget>
#include "ui_FilterWidget.h"
#include "BaseProcessWidget.h"

class FilterWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    FilterWidget(QWidget *parent = Q_NULLPTR);
    ~FilterWidget();
private slots:
    void on_btnRun_clicked();
    void on_cbFilterType_currentIndexChanged(int index);

private:
    Ui::FilterWidget    ui;
};
