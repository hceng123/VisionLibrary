#pragma once

#include <QWidget>
#include "ui_CCWidget.h"
#include "BaseProcessWidget.h"

class CCWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    CCWidget(QWidget *parent = Q_NULLPTR);
    ~CCWidget();

private:
    VisionStatus _runRemoveCC(cv::Mat &mat);

private slots:
    void on_btnRun_clicked();
    void on_btnApply_clicked();

private:
    Ui::CCWidget ui;
};
