@echo OFF

REM This file copies the libraries from the bjam stage directory
REM into the desired output directory. The first argument is the
REM --stagedir from the bjam command (i.e., without the trailing
REM ".../lib"). The second argument is the desired destination
REM directory for the libraries -- essentially the --libdir if
REM the bjam command was an "install". (2010-09-15 MM)

echo Copying Boost libraries...

set ARG_BOOST_STAGEDIR=%1
set ARG_BOOST_LIBDIR=%2

if "_%ARG_BOOST_STAGEDIR%_" == "__" (
    echo Script argument 1, bjam stagedir, missing.
    goto :bad )

if "_%ARG_BOOST_LIBDIR%_" == "__" (
    echo Script argument 2, bjam libdir, missing.
    goto :bad )

if not exist %ARG_BOOST_LIBDIR% (
    echo Creating output directory...
    mkdir %ARG_BOOST_LIBDIR%
    if ERRORLEVEL 1 (
        echo Script failed to create output folder.
        goto :bad ) )

copy /Y %ARG_BOOST_STAGEDIR%\lib\* %ARG_BOOST_LIBDIR% > nul

if ERRORLEVEL 1 (
    echo Script failed to copy libraries.
    goto :bad )

:end
echo Copying Boost libraries completed.
exit /b 0

:bad
echo Script exiting with errors.
exit /b 1
