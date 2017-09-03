#ifndef _TIME_LOG_H_
#define _TIME_LOG_H_

#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include "BaseType.h"
#include "StopWatch.h"
#include <atomic>

namespace AOI
{
namespace Vision
{

class TimeLog : private Uncopyable
{
protected:
    TimeLog(): _anIndex ( 0 ), _vecStringTimeLog ( _SIZE, "" ) {};
public:
    static TimeLog *GetInstance();
    void addTimeLog(const std::string &strMsg);
    void addTimeLog(const std::string &strMsg, __int64 nTimeSpan);
    void dumpTimeLog(const std::string &strFilePath);
private:
    static std::atomic<TimeLog*>    _pInstance;
    static std::mutex               _mutex;
    static const size_t             _SIZE   = 5000;

    std::atomic<size_t>             _anIndex;
    StringVector                    _vecStringTimeLog;
    CStopWatch                      _stopWatch;
};

}
}
#endif