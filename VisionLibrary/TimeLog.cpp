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
    String strLog = strMsg;
    strLog += "\t" + std::to_string( _stopWatch.Now() );
    strLog += "\t" + _stopWatch.GetLocalTimeStr();

    std::lock_guard<std::mutex> mutexTime(_mutexTimeLog);

    if ( _vecStringTimeLog.size() > 10000 )
        _vecStringTimeLog.clear();    
    _vecStringTimeLog.push_back(strLog);
}

void TimeLog::addTimeLog(const std::string &strMsg, __int64 nTimeSpan) {
    String strLog = strMsg;
    strLog += "\t" + std::to_string ( nTimeSpan );
    strLog += "\t" + _stopWatch.GetLocalTimeStr();

    std::lock_guard<std::mutex> mutexTime ( _mutexTimeLog );

    if ( _vecStringTimeLog.size () > 5000 )
        _vecStringTimeLog.clear ();
    
    _vecStringTimeLog.push_back ( strLog );
}

void TimeLog::dumpTimeLog(const std::string &strFilePath) {
    std::ofstream file(strFilePath);
    if ( ! file.is_open() )
        return;

    std::unique_lock<std::mutex> mutexTime(_mutexTimeLog);
    auto vecLocalTimeLog = _vecStringTimeLog;
    _vecStringTimeLog.clear();
    mutexTime.unlock();

    for( const auto &strLog : vecLocalTimeLog )
        file << strLog << std::endl;
    file.close();
}

}
}