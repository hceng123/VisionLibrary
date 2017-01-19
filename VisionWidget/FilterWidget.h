#pragma once

#include <QWidget>
#include "ui_FilterWidget.h"

class FilterWidget : public QWidget
{
    Q_OBJECT

public:
    FilterWidget(QWidget *parent = Q_NULLPTR);
    ~FilterWidget();

private slots:
    void on_btnRun_clicked();

private:
    Ui::FilterWidget ui;
};
