//Import from book << Windows Via C/C++ >>. SG.Xiao

#ifndef _STOP_WATCH_
#define _STOP_WATCH_
#pragma once

#include <chrono>
#include <string>
#include <time.h>
#include "boost/date_time.hpp"

using namespace std::chrono;
namespace pt = boost::posix_time;

class CStopWatch
{
public:
    CStopWatch() {
        Start();
    }

    void Start() {
        _tpStart = high_resolution_clock::now();
    }

    __int64 Now() const {
        high_resolution_clock::time_point tpNow = high_resolution_clock::now();
        return static_cast<__int64> ((std::chrono::duration<double, std::milli>(tpNow - _tpStart)).count());
    }

    __int64 Span() {
        high_resolution_clock::time_point tpNow = high_resolution_clock::now();
        auto span = static_cast<__int64> (std::chrono::duration<double, std::milli>(tpNow - _tpStart).count());
        _tpStart = tpNow;
        return span;
    }

    __int64 static AbsNow() {
        return duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }

    __int64 NowInMicro() const {
        high_resolution_clock::time_point tpNow = high_resolution_clock::now();
        return static_cast<__int64> ((std::chrono::duration<double, std::micro>(tpNow - _tpStart)).count());
    }

    __int64 SpanInMicro() {
        high_resolution_clock::time_point tpNow = high_resolution_clock::now();
        auto span = static_cast<__int64> (std::chrono::duration<double, std::micro>(tpNow - _tpStart).count());
        _tpStart = tpNow;
        return span;
    }

    __int64 static AbsNowInMicro() {
        return duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();
    }

    std::string static GetLocalTimeStr() {
        time_t nTimeSeconds;
        time(&nTimeSeconds);
        std::string strTime = pt::to_iso_string(pt::second_clock::local_time());
        __int64 nMilliseconds = AbsNow() - nTimeSeconds * 1000;
        strTime += "." + std::to_string(nMilliseconds);
        return strTime;
    }

private:
    high_resolution_clock::time_point _tpStart;
};

#endif