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

using namespace AOI;

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
    void on_houghCircleBtn_clicked();
    void on_findCircleBtn_clicked();
    void on_calcRoundnessBtn_clicked();
    void on_fitLineBtn_clicked();
    void on_btnFindLine_clicked();
    void on_fitParallelLineBtn_clicked();
    void on_fitRectBtn_clicked();
    void on_ocrBtn_clicked();
    void on_srchFiducialBtn_clicked();
    void on_btnLrnTemplate_clicked();
    void on_matchTmplBtn_clicked();
    void on_btnCalcCoverage_clicked();
    void on_btnCalibrateCamera_clicked();
    void on_btnLrnChip_clicked();
    void on_btnInspChip_clicked();
    void on_btnCountEdge_clicked();
    void on_btnLrnContour_clicked();
    void on_btnInspContour_clicked();
    void on_btnInspHole_clicked();
    void on_btnAutoLocateLead_clicked();
    void on_btnInspLead_clicked();
    void on_btnGridAvgGrayScale_clicked();
    void on_btnCameraMTF_clicked();    
protected:
    bool checkDisplayImage();
    void drawTmplImage(const cv::Mat &matTmpl);
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

    Vision::Int32                       _nChipRecordId = -1;
    Vision::Int32                       _nContourRecordId = -1;
    Vision::Int32                       _nTmplRecordId = -1;
};

#endif // VISIONWIDGET_H
