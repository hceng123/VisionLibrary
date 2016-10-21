#include "visionwidget.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    VisionWidget w;
    w.show();
    return a.exec();
}
