//Import from book << Windows Via C/C++ >>. SG.Xiao

#ifndef _STOP_WATCH_
#define _STOP_WATCH_
#pragma once

#include <chrono>
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

private:
	high_resolution_clock::time_point _tpStart;
};

#endif