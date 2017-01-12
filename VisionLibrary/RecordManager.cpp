#include "RecordManager.h"
#include "Log.h"
#include "boost/filesystem.hpp"
#include "Config.h"
#include "opencv2/core.hpp"
#include "VisionType.h"

namespace bfs = boost::filesystem;

namespace AOI
{
namespace Vision
{

/*static*/ RecordManager* RecordManager::getInstance()
{
    static RecordManager recordManager;
    return &recordManager;
}

RecordManager::~RecordManager()
{
}

VisionStatus RecordManager::add(IRecord *pRecord, Int32 &recordID)
{
    recordID = _generateRecordID();    
    pRecord->save ( _getRecordFilePath ( recordID ) );
    _mapRecord.insert ( PairRecord (recordID, pRecord) );

    String strLog = "Add record " + std::to_string(recordID) + ";";
    WriteLog(strLog);
    return VisionStatus::OK;
}

IRecord* RecordManager::get(Int32 nRecordID)
{
    return _mapRecord[nRecordID];
}

VisionStatus RecordManager::free(Int32 nRecordID)
{
    IRecord *pRecord = _mapRecord[nRecordID];
    if ( pRecord != nullptr )
    {
        _mapRecord.erase(nRecordID);
        delete pRecord;
    }
    
    bfs::remove( _getRecordFilePath(nRecordID) );
    
    String strLog = "Free record " + std::to_string(nRecordID) + ";";
    WriteLog(strLog);
    return VisionStatus::OK;
}

VisionStatus RecordManager::freeAllRecord()
{
    for ( auto record : _mapRecord )
    {
       delete record.second;
    }
    _mapRecord.clear();
    bfs::remove_all( Config::GetInstance()->getRecordDir() );
    WriteLog("All record is freed");
    return VisionStatus::OK;
}

VisionStatus RecordManager::load()
{
    auto strRecordDir = Config::GetInstance()->getRecordDir();
    if ( ! bfs::exists( strRecordDir ) )   {
        bfs::create_directories(strRecordDir);
        return VisionStatus::OK;
    }
    Int32Vector vecRecord;
    for ( auto &p : bfs::directory_iterator(strRecordDir)) {
        if ( ! bfs::is_directory ( p.status() ) )
        {
            _loadOneRecord ( p.path().string() );
        }
    }
    return VisionStatus::OK;
}

VisionStatus RecordManager::_loadOneRecord(const String &strFilePath)
{
    auto pos = strFilePath.rfind('.');
    if (pos != String::npos)  {
        auto strRecordID = strFilePath.substr(pos + 1);
        auto nRecordID = stoi(strRecordID);

        cv::FileStorage fs( strFilePath, cv::FileStorage::READ);
        cv::FileNode fileNode = fs["type"];
        Int32 recordType;
        cv::read(fileNode, recordType, static_cast<Int32> ( PR_RECORD_TYPE::INVALID ) );
        if ( static_cast<Int32> ( PR_RECORD_TYPE::INVALID ) == recordType ) {
            fs.release();
            return VisionStatus::INVALID_RECORD_TYPE;
        }
        IRecord *pRecord = _createRecordPtr ( recordType );
        if ( nullptr == pRecord )   {
            fs.release();
            return VisionStatus::INVALID_RECORD_TYPE;
        }
        pRecord->load(fs);
        fs.release();
        _mapRecord.insert(PairRecord(nRecordID, pRecord));
    }
    return VisionStatus::OK;
}

IRecord *RecordManager::_createRecordPtr(Int32 recordType)
{
    PR_RECORD_TYPE enRecordType = static_cast<PR_RECORD_TYPE>(recordType);
    switch(enRecordType)
    {
    case PR_RECORD_TYPE::ALIGNMENT:
        return ( new TmplRecord( enRecordType ) );
        break;
    default:
        return nullptr;
    }
    return nullptr;
}

Int32 RecordManager::_generateRecordID()
{
    const int       START_RECORD_ID =       1;
    auto strRecordDir = Config::GetInstance()->getRecordDir();
    if ( ! bfs::exists( strRecordDir ) )   {
        bfs::create_directories( strRecordDir );
        return START_RECORD_ID;
    }
    Int32Vector vecRecord;
    for ( auto &p : bfs::directory_iterator(strRecordDir)) {
        if ( ! bfs::is_directory ( p.status() ) ) 
        {
            String path = p.path().string();
            auto pos = path.rfind('.');
            if ( pos != String::npos )  {
                auto strRecordID = path.substr(pos + 1);
                auto nRecordID = stoi ( strRecordID );
                vecRecord.push_back(nRecordID);
            }
        }
    }
    std::sort(vecRecord.begin(), vecRecord.end());
    auto targetRecordID = START_RECORD_ID;
    for ( auto recordID : vecRecord )   {
        if ( recordID != targetRecordID )
            return targetRecordID;
        ++ targetRecordID;
    }

    return targetRecordID;
}

String RecordManager::_getRecordFilePath(Int32 recordID)
{
    char chArrRecordID[10];
    _snprintf( chArrRecordID, sizeof(chArrRecordID), "%02d", recordID );
    String strFilePath = Config::GetInstance()->getRecordDir() + Config::GetInstance()->getRecordPreFix() + String(chArrRecordID);
    return strFilePath;
}

}
}