#ifndef VISIONWIDGET_H
#define VISIONWIDGET_H

#include <QtWidgets/QMainWindow>
#include "ui_visionwidget.h"
#include "FilterWidget.h"
#include "ThresholdWidget.h"
#include "GrayScaleWidget.h"
#include "CCWidget.h"
#include "EdgeDetectWidget.h"
#include "FillHoleWidget.h"
#include "ColorWidget.h"

class VisionWidget : public QMainWindow
{
    Q_OBJECT

public:
    VisionWidget(QWidget *parent = 0);
    ~VisionWidget();
private slots:
    void on_selectImageBtn_clicked();
    void on_checkBoxByerFormat_clicked(bool checked);
	void on_fitCircleBtn_clicked();
    void on_calcRoundnessBtn_clicked();
    void on_fitLineBtn_clicked();
    void on_btnCaliper_clicked();
    void on_fitParallelLineBtn_clicked();
    void on_fitRectBtn_clicked();
    void on_ocrBtn_clicked();
    void on_srchFiducialBtn_clicked();
    void on_selectTemplateBtn_clicked();
    void on_matchTmplBtn_clicked();
    void on_captureTemplateBtn_clicked();
    void on_btnCalcCoverage_clicked();
    void on_btnCalibrateCamera_clicked();
    void on_btnLrnChip_clicked();
    void on_btnInspChip_clicked();
    void on_btnCountEdge_clicked();
    void on_btnLrnContour_clicked();
    void on_btnInspContour_clicked();
    void on_btnInspHole_clicked();
protected:
    bool checkDisplayImage();
    void drawTmplImage();
private:
    Ui::VisionWidgetClass               ui;
    std::string                         _sourceImagePath;
    cv::Mat                             _matOriginal;
    std::unique_ptr<QIntValidator>      _ptrIntValidator;
    std::unique_ptr<FilterWidget>       _ptrFilterWidget;
    std::unique_ptr<ThresholdWidget>    _ptrThresholdWidget;
    std::unique_ptr<GrayScaleWidget>    _ptrGrayScaleWidget;
    std::unique_ptr<CCWidget>           _ptrCcWidget;
    std::unique_ptr<FillHoleWidget>     _ptrFillHoleWidget;
    std::unique_ptr<EdgeDetectWidget>   _ptrEdgeDetectWidget;
    std::unique_ptr<ColorWidget>        _ptrColorWidget;
    cv::Mat                             _matTmpl;

    AOI::Int32                          _nChipRecordId = -1;
    AOI::Int32                          _nContourRecordId = -1;
};

#endif // VISIONWIDGET_H
