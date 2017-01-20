#pragma once

#include <QWidget>
#include "ui_FilterWidget.h"
#include "visionview.h"

class FilterWidget : public QWidget
{
    Q_OBJECT

public:
    FilterWidget(QWidget *parent = Q_NULLPTR);
    ~FilterWidget();
    void setVisionView(VisionView* pVisionView);
private slots:
    void on_btnRun_clicked();
    void on_cbFilterType_currentIndexChanged(int index);

private:
    Ui::FilterWidget    ui;
    VisionView*          _pVisionView;
};
