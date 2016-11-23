#ifndef VISIONWIDGET_H
#define VISIONWIDGET_H

#include <QtWidgets/QMainWindow>
#include "ui_visionwidget.h"

class VisionWidget : public QMainWindow
{
    Q_OBJECT

public:
    VisionWidget(QWidget *parent = 0);
    ~VisionWidget();

private slots:
    void on_selectImageBtn_clicked();
		void on_fitCircleBtn_clicked();
private:
    Ui::VisionWidgetClass ui;
};

#endif // VISIONWIDGET_H
