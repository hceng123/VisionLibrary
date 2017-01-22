#ifndef THRESHOLDWIDGET_H
#define THRESHOLDWIDGET_H

#include <QWidget>
#include "ui_ThresholdWidget.h"
#include "visionview.h"
#include "BaseProcessWidget.h"

class ThresholdWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    ThresholdWidget(QWidget *parent = 0);
    ~ThresholdWidget();

private slots:
    void on_btnAutoThreshold_clicked();
    void on_sliderThreshold_valueChanged(int position);
    void on_lineEditThreshold_returnPressed();
    void on_btnApply_clicked();

private:
    cv::Mat _runThreshold(int nThreshold);

private:
    Ui::ThresholdWidget             ui;
    std::unique_ptr<QIntValidator>  _ptrThresValidator;
};

#endif // THRESHOLDWIDGET_H
