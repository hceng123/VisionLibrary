#ifndef THRESHOLDWIDGET_H
#define THRESHOLDWIDGET_H
#include "ui_ThresholdWidget.h"
#include "BaseProcessWidget.h"

class ThresholdWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    ThresholdWidget(QWidget *parent = 0);
    ~ThresholdWidget();
    virtual std::string myName() const override;
private slots:
    void on_btnAutoThreshold_clicked();
    void on_sliderThreshold_valueChanged(int position);
    void on_lineEditThreshold_returnPressed();
    void on_checkBoxReverseThres_clicked();

private:
    cv::Mat _runThreshold(int nThreshold);

private:
    Ui::ThresholdWidget             ui;
    std::unique_ptr<QIntValidator>  _ptrThresValidator;
};

#endif // THRESHOLDWIDGET_H
