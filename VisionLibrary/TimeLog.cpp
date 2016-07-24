#include "TimeLog.h"
#include <fstream>

namespace AOI
{
namespace Vision
{

TimeLog *TimeLog::GetInstance()
{
    static TimeLog timeLog;
    return &timeLog;
}

void TimeLog::addTimeLog(const std::string &strMsg)
{
    std::lock_guard<std::mutex> mutexTime(_mutexTimeLog);

    if ( _vecStringTimeLog.size() > 5000 )
        _vecStringTimeLog.clear();
    std::string strLog = strMsg;
    strLog += "\t" + std::to_string( _stopWatch.Now() );
    strLog += "\t" + _stopWatch.GetLocalTimeStr();
    _vecStringTimeLog.push_back(strLog);
}

void TimeLog::addTimeLog(const std::string &strMsg, __int64 nTimeSpan)
{
     std::lock_guard<std::mutex> mutexTime(_mutexTimeLog);

    if ( _vecStringTimeLog.size() > 5000 )
        _vecStringTimeLog.clear();
    std::string strLog = strMsg;
    strLog += "\t" + std::to_string( nTimeSpan);
    _vecStringTimeLog.push_back(strLog);
}

void TimeLog::dumpTimeLog(const std::string &strFilePath)
{
    std::ofstream file(strFilePath);
    if ( ! file.is_open() )
        return;

    std::lock_guard<std::mutex> mutexTime(_mutexTimeLog);
    for( const auto &strLog : _vecStringTimeLog )
        file << strLog << std::endl;

    _vecStringTimeLog.clear();
    file.close();
}

}
}