#include "visionwidget.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QFont font;
	font.setFamily("MS Shell Dlg 2");
	font.setPixelSize(12);
	a.setFont(font);
    VisionWidget w;
    w.show();
    return a.exec();
}
