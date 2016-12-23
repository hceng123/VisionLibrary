#ifndef DIALOGVISIONTOOLBOX_H
#define DIALOGVISIONTOOLBOX_H

#include <QDialog>
#include "VisionAPI.h"
#include "visionview.h"

namespace Ui {
class DialogVisionToolBox;
}

class DialogVisionToolBox : public QDialog
{
    Q_OBJECT

public:
    explicit DialogVisionToolBox(QWidget *parent = 0, VisionView *pVisionView = 0);
    ~DialogVisionToolBox();

private slots:
    void on_comboBoxDebugMode_currentIndexChanged(int index);
    void on_btnDumpTimeLog_clicked();
    void on_btnCheckIntensity_clicked();

private:
    Ui::DialogVisionToolBox *ui;
    VisionView *_pVisionView;
};

#endif // DIALOGVISIONTOOLBOX_H
