//Import from book << Windows Via C/C++ >>. SG.Xiao

#ifndef _WIN_STOP_WATCH_
#define _WIN_STOP_WATCH_
#pragma once
#include "stdafx.h"
#include <Windows.h>

class CWinStopWatch
{
public:
	CWinStopWatch()
	{
		QueryPerformanceFrequency( &m_liPerfFreq );
		Start();
	}
	void Start() { QueryPerformanceCounter(&m_liPerfStart);}

	__int64 Now () const
	{
		LARGE_INTEGER liPerfNow;
		QueryPerformanceCounter ( &liPerfNow );
		return ( ( ( liPerfNow.QuadPart - m_liPerfStart.QuadPart ) * 1000) / m_liPerfFreq.QuadPart);
	}

	__int64 AbsNow () const
	{
		LARGE_INTEGER liPerfNow;
		QueryPerformanceCounter ( &liPerfNow );
		return ( ( liPerfNow.QuadPart * 1000 ) / m_liPerfFreq.QuadPart);
	}

	__int64 NowInMicro () const
	{
		LARGE_INTEGER liPerfNow;
		QueryPerformanceCounter ( &liPerfNow );
		return ( ( ( liPerfNow.QuadPart - m_liPerfStart.QuadPart ) * 1000000) / m_liPerfFreq.QuadPart);
	}

	__int64 AbsNowInMicro () const
	{
		LARGE_INTEGER liPerfNow;
		QueryPerformanceCounter ( &liPerfNow );
		return ( ( liPerfNow.QuadPart * 1000000 ) / m_liPerfFreq.QuadPart);
	}

private:
	LARGE_INTEGER m_liPerfFreq;
	LARGE_INTEGER m_liPerfStart;
};

#endif