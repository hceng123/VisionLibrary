#ifndef VISIONVIEWTOOLBOX_H
#define VISIONVIEWTOOLBOX_H

#include <QWidget>
#include <visionview.h>

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
    void on_zoomInButton_clicked();
    void on_zoomOutButton_clicked();
    void on_toolBoxPushButton_clicked();

private:
    Ui::VisionViewToolBox *ui;
    VisionView *_pVisionView;
};

#endif // VISIONVIEWTOOLBOX_H
