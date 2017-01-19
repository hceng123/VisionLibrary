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
    void on_fitLineBtn_clicked();
    void on_fitParallelLineBtn_clicked();
    void on_fitRectBtn_clicked();
    void on_ocrBtn_clicked();
    void on_srchFiducialBtn_clicked();
    void on_addPreProcessorBtn_clicked();
    void on_checkBoxDisplayGrayScale_clicked(bool checked);
    void on_checkBoxDisplayBinary_clicked(bool checked);
    void on_checkBoxReverseThres_clicked(bool checked);
    void on_sliderThreshold_valueChanged(int position);
    void on_lineEditBinaryThreshold_returnPressed();
    void on_lineEditRRatio_returnPressed();
    void on_lineEditGRatio_returnPressed();
    void on_lineEditBRatio_returnPressed();

protected:
    bool checkDisplayImage();
private:
    Ui::VisionWidgetClass               ui;
    std::string                         _sourceImagePath;
    cv::Mat                             _matOriginal;
    std::unique_ptr<QIntValidator>      _ptrIntValidator;
    std::unique_ptr<QDoubleValidator>   _ptrDoubleValidator;
    std::unique_ptr<QIntValidator>      _ptrThresValidator;
};

#endif // VISIONWIDGET_H
