#include "RecordManager.h"
#include "Log.h"
#include "boost/filesystem.hpp"
#include "Config.h"
#include "opencv2/core.hpp"
#include "VisionType.h"
#include "FileUtils.h"
#include "JoinSplit.h"

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

RecordManager::~RecordManager() {
}

VisionStatus RecordManager::add(RecordPtr pRecord, Int32 &nRecordId) {
    nRecordId = _generateRecordID();
    String strFilePath = _getRecordFilePath(nRecordId);
    FileUtils::MakeDirectory(strFilePath);
    pRecord->save(strFilePath);
    joinDir(strFilePath.c_str(), Config::GetInstance()->getRecordExt().c_str());
    FileUtils::RemoveAll(strFilePath);
    if (_mapRecord.find(nRecordId) != _mapRecord.end()) {
        String strLog = "Record " + std::to_string(nRecordId) + " already exist in memory, this is a serious problem.";
        WriteLog(strLog);
    }
    else {
        _mapRecord.insert(std::make_pair(nRecordId, pRecord));
        String strLog = "Add record " + std::to_string(nRecordId) + ";";
        WriteLog(strLog);
    }
    return VisionStatus::OK;
}

RecordPtr RecordManager::get(Int32 nRecordId) {
    if (_mapRecord.find(nRecordId) == _mapRecord.end()) {
        String strLog = "Cannot find record id " + std::to_string(nRecordId) + ".";
        WriteLog(strLog);
        return nullptr;
    }

    return _mapRecord[nRecordId];
}

VisionStatus RecordManager::free(Int32 nRecordId) {
    RecordPtr pRecord = _mapRecord[nRecordId];
    if ( pRecord != nullptr )
        _mapRecord.erase(nRecordId);
    
    String strRecordFilePath = _getRecordFilePath ( nRecordId ) + Config::GetInstance()->getRecordExt();
    FileUtils::Remove ( strRecordFilePath );
    
    String strLog = "Free record " + std::to_string ( nRecordId ) + ";";
    WriteLog(strLog);
    return VisionStatus::OK;
}

VisionStatus RecordManager::freeAllRecord() {
    _mapRecord.clear();
    FileUtils::ClearFolder(Config::GetInstance()->getRecordDir());
    WriteLog("All record is freed.");
    return VisionStatus::OK;
}

VisionStatus RecordManager::load() {
    auto strRecordDir = Config::GetInstance()->getRecordDir();
    if (! bfs::exists(strRecordDir)) {
        bfs::create_directories(strRecordDir);
        return VisionStatus::OK;
    }
    VisionStatus enStatus = VisionStatus::OK;
    for (auto &p : bfs::directory_iterator(strRecordDir)) {
        if (bfs::is_regular_file(p.status())) {
            enStatus = _loadOneRecord(p.path().string());
            if (enStatus != VisionStatus::OK)
                return enStatus;
        }
    }
    return enStatus;
}

VisionStatus RecordManager::_loadOneRecord(const String &strFilePath) {
    auto pos = strFilePath.rfind('.');
    if (pos != String::npos) {
        auto strRecordID = strFilePath.substr(pos - 3, 3);
        auto nRecordId = stoi(strRecordID);
        String strTargetDir = Config::GetInstance()->getRecordDir() + strRecordID;
        if (! FileUtils::Exists(strTargetDir))
            FileUtils::MakeDirectory(strTargetDir);
        splitFiles(strFilePath.c_str(), strTargetDir.c_str());

        String strParamFile = strTargetDir + "/" + Config::GetInstance()->getRecordParamFile();
        cv::FileStorage fs(strParamFile, cv::FileStorage::READ);
        cv::FileNode fileNode = fs["type"];
        Int32 recordType;
        cv::read(fileNode, recordType, static_cast<Int32> (PR_RECORD_TYPE::INVALID));
        if (static_cast<Int32> (PR_RECORD_TYPE::INVALID) == recordType) {
            fs.release();
            FileUtils::RemoveAll(strTargetDir);
            return VisionStatus::INVALID_RECORD_TYPE;
        }
        RecordPtr pRecord = _createRecordPtr(recordType);
        if (nullptr == pRecord) {
            fs.release();
            FileUtils::RemoveAll(strTargetDir);
            return VisionStatus::INVALID_RECORD_TYPE;
        }
        pRecord->load(fs, strTargetDir);
        fs.release();
        _mapRecord.insert(PairRecord(nRecordId, pRecord));
        FileUtils::RemoveAll(strTargetDir);
    }
    return VisionStatus::OK;
}

RecordPtr RecordManager::_createRecordPtr(Int32 recordType) {
    PR_RECORD_TYPE enRecordType = static_cast<PR_RECORD_TYPE>(recordType);
    switch (enRecordType) {
    case PR_RECORD_TYPE::OBJECT:
        return std::make_shared<ObjRecord>(enRecordType);
        break;
    case PR_RECORD_TYPE::DEVICE:
        return std::make_shared<DeviceRecord>(enRecordType);
        break;
    case PR_RECORD_TYPE::CHIP:
        return std::make_shared<ChipRecord>(enRecordType);
        break;
    case PR_RECORD_TYPE::CONTOUR:
        return std::make_shared<ContourRecord>(enRecordType);
        break;
    case PR_RECORD_TYPE::TEMPLATE:
        return std::make_shared<TmplRecord>(enRecordType);
        break;
    case PR_RECORD_TYPE::OCV:
        return std::make_shared<OcvRecord>(enRecordType);
        break;
    default:
        assert(0);
    }
    return nullptr;
}

Int32 RecordManager::_generateRecordID() {
    const int START_RECORD_ID = 1;
    auto strRecordDir = Config::GetInstance()->getRecordDir();
    if (! bfs::exists(strRecordDir)) {
        bfs::create_directories(strRecordDir);
        return START_RECORD_ID;
    }
    Int32Vector vecRecord;
    for (auto &p : bfs::directory_iterator(strRecordDir)) {
        if (! bfs::is_directory(p.status())) {
            String path = p.path().string();
            auto pos = path.rfind('.');
            if (pos != String::npos) {
                auto strRecordID = path.substr(pos - 3, 3);
                auto nRecordId = stoi(strRecordID);
                vecRecord.push_back(nRecordId);
            }
        }
    }
    std::sort(vecRecord.begin(), vecRecord.end());
    auto targetRecordID = START_RECORD_ID;
    for (auto recordID : vecRecord) {
        if (recordID != targetRecordID)
            return targetRecordID;
        ++ targetRecordID;
    }

    return targetRecordID;
}

String RecordManager::_getRecordFilePath(Int32 recordID) {
    char chArrRecordID[10];
    _snprintf(chArrRecordID, sizeof(chArrRecordID), "%03d", recordID);
    String strFilePath = Config::GetInstance()->getRecordDir() + String(chArrRecordID);
    return strFilePath;
}

}
}