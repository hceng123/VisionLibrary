#ifndef GRAYSCALEWIDGET_H
#define GRAYSCALEWIDGET_H

#include <QWidget>
#include "ui_GrayScaleWidget.h"
#include "BaseProcessWidget.h"

class GrayScaleWidget : public BaseProcessWidget
{
    Q_OBJECT

public:
    GrayScaleWidget(QWidget *parent = 0);
    ~GrayScaleWidget();
    virtual QSize	sizeHint() const;
    virtual std::string MyName() const override;
private slots:
    void on_btnConvert_clicked();
    void on_btnApply_clicked();

private:
    cv::Mat _convertImage();

private:
    Ui::GrayScaleWidget ui;
};

#endif // GRAYSCALEWIDGET_H
