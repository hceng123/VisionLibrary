#pragma once

#include "ui_CCWidget.h"
#include "BaseProcessWidget.h"

class CCWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    CCWidget(QWidget *parent = Q_NULLPTR);
    ~CCWidget();
    virtual std::string myName() const override;
private:
    VisionStatus _runRemoveCC(cv::Mat &mat);

private slots:
    void on_btnRun_clicked();

private:
    Ui::CCWidget ui;
};
