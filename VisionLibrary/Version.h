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

//2017-09-23 (XSG) Use Nelder-Mead algorithm to do auto threshold to speed up.
//2017-09-23 (XSG) Add PR_Calc3DBase function to avoid repeatly calculate base surface.
#define AOI_VISION_VERSION          "1.00.07"

//2017-09-17 (XSG) Add PR_Comb3DCalib function.
//2017-09-16 (XSG) Add bSubPixelRefine in PR_MATCH_TEMPLATE_CMD. Default is false.
//#define AOI_VISION_VERSION          "1.00.06"

//2017-09-16 (XSG) Support 5 steps calibration.
//2017-09-16 (XSG) Use new method to do 3D calibrate, remove the top error points to fit surface.
//#define AOI_VISION_VERSION          "1.00.05"

//2017-09-15 (XSG) Add PR_Calc3DHeightDiff function.
//2017-09-14 (XSG) Enhance the Unwrap::_phaseUnwrapSurfaceTrk, add procedure unwrap from bottom to top.
//#define AOI_VISION_VERSION          "1.00.04"

//2017-09-14 (XSG) Add CALIB_3D_HEIGHT_SURFACE_TOO_SMALL status for calib 3D height.
//2017-09-14 (XSG) Remove 4 times harmonic wave.
//2017-09-14 (XSG) Fix Unwrap::_getIntegralTree not finish yet problem. Handle standalone pole points.
//#define AOI_VISION_VERSION          "1.00.03"

//2017-09-13 (XSG) Add PR_CalcPD function to calculate system pattern distortion.
//#define AOI_VISION_VERSION          "1.00.01"