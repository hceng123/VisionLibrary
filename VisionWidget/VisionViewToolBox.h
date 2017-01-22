#ifndef VISIONVIEWTOOLBOX_H
#define VISIONVIEWTOOLBOX_H

#include <QWidget>
#include "visionview.h"
#include "dialogvisiontoolbox.h"
#include <memory>

namespace Ui {
class VisionViewToolBox;
}

class VisionViewToolBox : public QWidget
{
    Q_OBJECT

public:
    explicit VisionViewToolBox(QWidget *parent = 0);
    ~VisionViewToolBox();
    void SetVisionView(VisionView *pVisionView);

private slots:
    void on_swapImageButton_clicked();
    void on_zoomInButton_clicked();
    void on_zoomOutButton_clicked();
    void on_toolBoxButton_clicked();
    void on_saveImageButton_clicked();
private:
    Ui::VisionViewToolBox                   *ui;
    VisionView                              *_pVisionView;
    std::unique_ptr<DialogVisionToolBox>    _pVisionToolDialog;
};

#endif // VISIONVIEWTOOLBOX_H
