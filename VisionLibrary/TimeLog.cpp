#include "TimeLog.h"
#include <fstream>

namespace AOI
{
namespace Vision
{

std::atomic<TimeLog*> TimeLog::_pInstance = nullptr;
std::mutex            TimeLog::_mutex;

TimeLog *TimeLog::GetInstance() {
    if ( _pInstance.load() == nullptr ) {
        std::lock_guard<std::mutex> lock( _mutex );
        if( _pInstance.load() == nullptr) {
            _pInstance = new TimeLog();
        }
    }
    return _pInstance;
}

void TimeLog::addTimeLog(const std::string &strMsg) {
    String strLog = strMsg;
    strLog += "\t" + std::to_string( _stopWatch.Span() );
    strLog += "\t" + _stopWatch.GetLocalTimeStr();

    auto nIndex = ( _anIndex ++ ) % _SIZE;
    _vecStringTimeLog [ nIndex ] = strLog;
}

void TimeLog::addTimeLog(const std::string &strMsg, __int64 nTimeSpan) {
    String strLog = strMsg;
    if ( strLog.size() < _MSG_WIDTH )
        strLog.resize(_MSG_WIDTH, ' ');
    strLog += "\t" + std::to_string ( nTimeSpan );
    strLog += "\t" + _stopWatch.GetLocalTimeStr();

    auto nIndex = ( _anIndex ++ ) % _SIZE;
    _vecStringTimeLog [ nIndex ] = strLog;
}

void TimeLog::dumpTimeLog(const std::string &strFilePath) {
    std::ofstream file(strFilePath);
    if ( ! file.is_open() )
        return;

    auto nIndex = ( _anIndex ) % _SIZE;
    auto vecLocalTimeLog = _vecStringTimeLog;
    
    for ( auto i = nIndex; i < _SIZE; ++ i ) {
        if ( !_vecStringTimeLog[i].empty () )
            file << _vecStringTimeLog[i] << std::endl;
    }

    for ( size_t i = 0; i < nIndex; ++ i ) {
        if ( !_vecStringTimeLog[i].empty () )
            file << _vecStringTimeLog[i] << std::endl;
    }
    
    file.close();
}

}
}