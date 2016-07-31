#ifndef _FILEUTILS_H_
#define _FILEUTILS_H_
//****************************************************************************
// FileUtils.h -- $Id$
//
// Purpose
//   Provides a set of file-handling utilities that are not included in
//   "boost::filesystem" as well as forwarders for some that are.
//
// Indentation
//   Four characters. No tabs!
//
// Modifications
//   2014-03-12 (MM) Added a temp file path generator.
//   2012-10-10 (MM) Added file appenders.
//   2011-01-14 (MM) Added file writers.
//   2010-12-20 (MM) Created.
//
// Copyright (c) 2010-2012, 2014 Agilent Technologies, Inc.  All rights reserved.
//****************************************************************************

#include "BaseDefs.h"

namespace AOI
{
namespace Vision
{
    class FileUtils
    {
        // Disable construction, destruction, etc.
        FileUtils();
        FileUtils(FileUtils const &);
        FileUtils &operator=(FileUtils const &);
       ~FileUtils();

    public:
        static bool Exists       (String const &pathname);
        static bool IsDirectory  (String const &pathname);
        static bool IsRegularFile(String const &pathname);

        static String GetParentPathname(String const &pathname);
        static String GetFilename      (String const &pathname);
        static String GetStem          (String const &pathname);
        static String GetExtension     (String const &pathname);

        static String AppendPaths(String const & lhs, String const & rhs);

        static void ReadBinaryFile(String const &filePathname, Binary &data);
        static void ReadStringFile(String const &filePathname, String &data);

        static void WriteBinaryFile(String const &filePathname, Binary const &data);
        static void WriteStringFile(String const &filePathname, String const &data);

        static void AppendToBinaryFile(String const &filePathname, Binary const &data);
        static void AppendToStringFile(String const &filePathname, String const &data);

        static void MakeDirectory(String const &directoryPathname);
        static void Remove       (String const &pathname);
        static void RemoveAll    (String const &directoryPathname);
        static void FileCopy     (String const &fromPathname, String const &intoDirectoryPathname, bool const writable=true);
        static void FileDuplicate(String const &fromPathname, String const &intoDirectoryPathname);
        static void Rename       (String const &fromPathname, String const &toPathname);

        static String GetTemporaryDirectoryPathname(String const &applicationSubPathname=String());
        static String GetTemporaryFilePathname     (String const &extension             =String(),    // Examples of an extension include SL("bsdl") and SL("txt").
                                                    String const &applicationSubPathname=String());
        static String GetPathInInstall             (String const &relativePathname      =String());

        static void GetPathsInDirectory(String const &directoryPathname, StringVector &paths);
    };
}
}
#endif//AGILENT_BBOX_COMMON_FILEUTILS_H
