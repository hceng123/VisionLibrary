/*****************************************************************************
 * Version.h -- $Id$
 *
 * Purpose
 *   This file includes the version of the library and the change history.
 *
 * Indentation
 *   Four characters. No tabs!
 *
 * Modifications
 *   2017-09-13 (XSG) Add PR_CalcPD function to calculate system pattern distortion.
 *   2016-10-14 (XSG) Created.
 *
 * Copyright (c) 2016-2016, Xiao Shengguang.  All rights reserved.
 ****************************************************************************/

//2017-09-14 (XSG) Add CALIB_3D_HEIGHT_SURFACE_TOO_SMALL status for calib 3D height.
//2017-09-14 (XSG) Remove 4 times harmonic wave.
//2017-09-14 (XSG) Fix Unwrap::_getIntegralTree not finish yet problem. Handle standalone pole points.
#define AOI_VISION_VERSION          "1.00.03"

//2017-09-13 (XSG) Add PR_CalcPD function to calculate system pattern distortion.
//#define AOI_VISION_VERSION          "1.00.01"