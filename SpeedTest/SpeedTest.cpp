// SpeedTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TestFunction.h"
#include "../VisionLibrary/VisionAPI.h"
#include <iostream>

using namespace AOI::Vision;

int _tmain(int argc, _TCHAR* argv[])
{
    std::cout << "START VISION LIBRARY SPEED TEST, TIME UNIT: ms" << std::endl;

    MatchTmplSpeedTest();

    TestTransposeSpeed();

    PR_FreeAllRecord();
	return 0;
}

