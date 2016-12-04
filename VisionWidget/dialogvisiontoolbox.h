#ifndef DIALOGVISIONTOOLBOX_H
#define DIALOGVISIONTOOLBOX_H

#include <QDialog>
#include "VisionAPI.h"

namespace Ui {
class DialogVisionToolBox;
}

class DialogVisionToolBox : public QDialog
{
    Q_OBJECT

public:
    explicit DialogVisionToolBox(QWidget *parent = 0);
    ~DialogVisionToolBox();

private slots:
    void on_comboBoxDebugMode_currentIndexChanged(int index);

private:
    Ui::DialogVisionToolBox *ui;
};

#endif // DIALOGVISIONTOOLBOX_H
