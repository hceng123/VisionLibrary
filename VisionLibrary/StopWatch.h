//Import from book << Windows Via C/C++ >>. SG.Xiao

#ifndef _STOP_WATCH_
#define _STOP_WATCH_
#pragma once

#include <chrono>
#include <string>
#include <time.h>
using namespace std::chrono;

class CStopWatch
{
public:
	CStopWatch()
	{
		Start();
	}

	void Start() 
    { 
        _tpStart = high_resolution_clock::now();
    }

	__int64 Now () const
	{
		high_resolution_clock::time_point tpNow = high_resolution_clock::now();
		return static_cast<__int64> ( std::chrono::duration<double, std::milli>( tpNow - _tpStart ).count() );
	}

	__int64 AbsNow () const
	{
		return duration_cast<milliseconds>( high_resolution_clock::now().time_since_epoch() ).count();
	}

	__int64 NowInMicro () const
	{
		high_resolution_clock::time_point tpNow = high_resolution_clock::now();
		return static_cast<__int64> ( std::chrono::duration<double, std::micro>( tpNow - _tpStart).count() );
	}

	__int64 AbsNowInMicro () const
	{
		return duration_cast<microseconds>( high_resolution_clock::now().time_since_epoch() ).count();
	}

    std::string GetLocalTimeStr()
    {
		char szTime[100];
        struct tm  stTm;
        time_t     nTimeSeconds;
        time(&nTimeSeconds);
        errno_t err = localtime_s( &stTm, &nTimeSeconds);
        __int64 nMilliseconds = AbsNow () - nTimeSeconds * 1000;
        _snprintf_s ( szTime, 100, "%02d:%02d:%02d.%03d", stTm.tm_hour, stTm.tm_min, stTm.tm_sec, nMilliseconds );
        return std::string(szTime);
    }
private:
	high_resolution_clock::time_point _tpStart;
};

#endif