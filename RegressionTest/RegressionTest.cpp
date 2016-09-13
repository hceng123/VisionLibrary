// RegressionTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TestFunction.h"

int _tmain(int argc, _TCHAR* argv[])
{
    /*************************
    * Test Device Inspection *
    **************************/
    TestInspDevice();
    TestInspDeviceAutoThreshold();

    /*************************
    * Test Device Inspection *
    **************************/
    TestRunLogcase();

    /*************************
    * Test Template Matching *
    **************************/
    TestTmplMatch();
    TestTmplMatch_Circle();

	return 0;
}

