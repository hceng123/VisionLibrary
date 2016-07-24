#ifndef _TIME_LOG_H_
#define _TIME_LOG_H_


#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include "BaseType.h"
#include "StopWatch.h"

namespace AOI
{
namespace Vision
{

class TimeLog:private Uncopyable
{
protected:
    TimeLog()   {};
public:
    static TimeLog *GetInstance();
    void addTimeLog(const std::string &strMsg);
    void addTimeLog(const std::string &strMsg, __int64 nTimeSpan);
    void dumpTimeLog(const std::string &strFilePath);
protected:
    std::mutex      _mutexTimeLog;
    StringVector    _vecStringTimeLog;
    CStopWatch      _stopWatch;
};

}
}
#endif