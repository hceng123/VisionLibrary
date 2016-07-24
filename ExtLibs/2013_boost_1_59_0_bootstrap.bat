@ECHO OFF

REM This file was derived from the "bootstrap.bat" file that
REM resides in the root of the Boost 1.57.0 distribution. It
REM has two arguments: a root folder (e.g., "boost_1_57_0")
REM and a toolset (e.g., "vc11", "vc12", or "vc13"). Both are
REM mandatory. If lucky, this file can be used for several
REM distributions. (2014-11-17 MM).
REM
REM Copyright (C) 2009 Vladimir Prus
REM
REM Distributed under the Boost Software License, Version 1.0.
REM (See accompanying file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)

ECHO Building Boost.Build engine

set ARG_BOOST_FOLDER=%1
set ARG_BOOST_TOOLSET=%2

if "_%ARG_BOOST_FOLDER%_" == "__" (
    echo Script argument 1, boost folder, missing. Example: boost_1_57_0.
    goto :bad )

if "_%ARG_BOOST_TOOLSET%_" == "__" (
    echo Script argument 2, boost toolset, missing. Example: vc9, vc10, vc11, vc12, etc.
    goto :bad )

if exist ".\%ARG_BOOST_FOLDER%\b2.exe" (
    echo Cancelling bootstrap build -- b2.exe already exists.
    goto :end )

if exist ".\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\b2.exe"      del .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\b2.exe
if exist ".\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\bjam.exe"    del .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\bjam.exe
if exist ".\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\b2.exe"   del .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\b2.exe
if exist ".\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\bjam.exe" del .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\bjam.exe

pushd .\%ARG_BOOST_FOLDER%\tools\build\src\engine

call .\build.bat %ARG_BOOST_TOOLSET% > ..\..\..\..\bootstrap.log
@ECHO OFF

popd

if exist ".\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\bjam.exe" (
   copy .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\b2.exe   .\%ARG_BOOST_FOLDER%\. > nul
   copy .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86\bjam.exe .\%ARG_BOOST_FOLDER%\. > nul
   goto :bjam_built)

if exist ".\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\bjam.exe" (
   copy .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\b2.exe   .\%ARG_BOOST_FOLDER%\. > nul
   copy .\%ARG_BOOST_FOLDER%\tools\build\src\engine\bin.ntx86_64\bjam.exe .\%ARG_BOOST_FOLDER%\. > nul
   goto :bjam_built)

goto :bjam_failure

:bjam_built

ECHO.
ECHO Bootstrapping is done. To build, run:
ECHO.
ECHO     .\%ARG_BOOST_FOLDER%\b2
ECHO.    
ECHO To adjust configuration, edit 'project-config.jam'.
ECHO Further information:
ECHO.
ECHO     - Command line help:
ECHO     .\%ARG_BOOST_FOLDER%\b2 --help
ECHO.     
ECHO     - Getting started guide: 
ECHO     http://boost.org/more/getting_started/windows.html
ECHO.     
ECHO     - Boost.Build documentation:
ECHO     http://www.boost.org/boost-build2/doc/html/index.html

goto :end

:bjam_failure

ECHO.
ECHO Failed to build Boost.Build engine.
ECHO Please consult bootstrap.log for furter diagnostics.
exit /b 2

:bad
echo Script exiting with errors.
exit /b 1

:end
exit /b 0
