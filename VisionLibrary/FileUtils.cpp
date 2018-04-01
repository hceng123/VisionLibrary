//****************************************************************************
// FileUtils.cpp -- $Id$
//
// Purpose
//   Implements a set of file-handling utilities that are not included
//   in "boost::filesystem" as well as forwarders for some that are.
//
// Indentation
//   Four characters. No tabs!
//
// Modifications
//   2014-03-12 (MM) Added a temp file path generator.
//   2012-11-16 (MM) Adapted to boost 1.52.0 changes.
//   2012-10-10 (MM) Added file appenders.
//   2011-01-14 (MM) Added file writers.
//   2011-01-10 (MM) Started using "file_string()" rather than "filename()".
//   2010-12-20 (MM) Created.
//
// Copyright (c) 2010-2012, 2014 Agilent Technologies, Inc.  All rights reserved.
//****************************************************************************

#include "BaseDefs.h"
#include "FileUtils.h"
#include <fstream>
#include <ios>
#include <limits>
#include <cstdio>
#define NOMINMAX                     // Inhibit definition of the MIN and MAX macros (from windows.h)
#include <windows.h>

namespace bfs = boost::filesystem;

#define SL(s) s

namespace AOI
{
namespace Vision
{
    namespace
    {
        String const FMT_NOT_FOUND            (SL("File does not exist: \"%s\"."));
        String const FMT_ALREADY_EXISTS       (SL("File already exists: \"%s\"."));
        String const FMT_NOT_REGULAR          (SL("File is not a regular file: \"%s\"."));
        String const FMT_NOT_DIRECTORY        (SL("File is not a directory file: \"%s\"."));
        String const FMT_TOO_LARGE            (SL("File is too large: \"%s\"."));
        String const FMT_NOT_OPENED_FOR_READ  (SL("File cannot be opened for read access: \"%s\"."));
        String const FMT_NOT_OPENED_FOR_WRITE (SL("File cannot be opened for write access: \"%s\"."));
        String const FMT_NOT_OPENED_FOR_APPEND(SL("File cannot be opened for append-write access: \"%s\"."));

        String MakeMessage(String const &format, bfs::path const &path)
        {
            return (boost::format(format) % path.string()).str();
        }

        Exception::Attributes MakeAttributes(bfs::path const &path)
        {
            Exception::Attributes attributes;
            attributes.push_back(Exception::Attribute(Exception::Runtime::attrType, SL("file")));
            attributes.push_back(Exception::Attribute(Exception::Runtime::attrPath, path.string()));
            return attributes;
        }

        String GetInitialPathname(void)
        {
            size_t const SIZE(1024);
            std::vector<char> cstr(SIZE, '\0');
            auto const rc(GetModuleFileNameA(nullptr, cstr.data(), SIZE));

            if (rc >= SIZE)  throw Exception::InternalError(SL("FileUtils::GetInitialPathname: Initial pathname too long."));

            return String(cstr.begin(), cstr.end());
        }

        String GetEnvironmentValue(String const & name)
        {
            size_t size(0u);

            {
                errno_t const returnCode(getenv_s(&size, nullptr, 0u, name.c_str()));

                if ((returnCode != 0) && (returnCode != ERANGE))  throw Exception::InternalError(SL("FileUtils::InEnvironment: Unexpected error returned by getenv_s."));

                if (size == 0u)  throw Exception::InternalError(SL("FileUtils::InEnvironment: Specified name does not exist in environment."));
            }

            std::vector<char> cstr(size, '\0');

            {
                errno_t const returnCode(getenv_s(&size, cstr.data(), size, name.c_str()));

                if (returnCode != 0)  throw Exception::InternalError(SL("FileUtils::InEnvironment: Unexpected error returned by getenv_s."));
            }

            return String(cstr.begin(), cstr.end() - 1);
        }
    }

    /*static*/bool FileUtils::Exists(String const &pathname)
    {
        return bfs::exists(bfs::path(pathname));
    }

    /*static*/bool FileUtils::IsDirectory(String const &pathname)
    {
        return bfs::is_directory(bfs::path(pathname));
    }

    /*static*/bool FileUtils::IsRegularFile(String const &pathname)
    {
        return bfs::is_regular_file(bfs::path(pathname));
    }

    /*static*/String FileUtils::GetParentPathname(String const &pathname)
    {
        return  bfs::path(pathname).parent_path().string();
    }

    /*static*/String FileUtils::GetFilename(String const &pathname)
    {
        return bfs::path(pathname).filename().string();
    }

    /*static*/String FileUtils::GetStem(String const &pathname)
    {
        return bfs::path(pathname).stem().string();
    }

    /*static*/String FileUtils::GetExtension(String const &pathname)
    {
        return bfs::path(pathname).extension().string();
    }

    /*static*/String FileUtils::AppendPaths(String const & lhs, String const & rhs)
    {
        return (bfs::path(lhs) / bfs::path(rhs)).string();
    }

    /*static*/void FileUtils::ReadBinaryFile(String const &filePathname, Binary &data)
    {
        bfs::path path = filePathname;

        if (!bfs::exists(path))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, path), MakeAttributes(path));

        if (!bfs::is_regular_file(path))
            throw Exception::InvalidArgument(MakeMessage(FMT_NOT_REGULAR, path), MakeAttributes(path));

        boost::uintmax_t reserve_size = bfs::file_size(path);

        if (reserve_size > std::numeric_limits<Binary::size_type>::max())
            throw Exception::InvalidOperation(MakeMessage(FMT_TOO_LARGE, path), MakeAttributes(path));

        Binary().swap(data);
        data.reserve(static_cast<Binary::size_type>(reserve_size));

        std::ifstream fs(path.string().c_str(), std::ios_base::in | std::ios_base::binary);

        if (!fs)
            throw Exception::ItemNotOpened(MakeMessage(FMT_NOT_OPENED_FOR_READ, path), MakeAttributes(path));

        for (std::ifstream::char_type c; fs.get(c);)
            data.push_back(c);

        fs.close();
    }

    /*static*/void FileUtils::ReadStringFile(String const &filePathname, String &data)
    {
        bfs::path path = filePathname;

        if (!bfs::exists(path))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, path), MakeAttributes(path));

        if (!bfs::is_regular_file(path))
            throw Exception::InvalidArgument(MakeMessage(FMT_NOT_REGULAR, path), MakeAttributes(path));

        boost::uintmax_t reserve_size = bfs::file_size(path);

        if (reserve_size > std::numeric_limits<Binary::size_type>::max())
            throw Exception::InvalidOperation(MakeMessage(FMT_TOO_LARGE, path), MakeAttributes(path));

        String().swap(data);
        data.reserve(static_cast<String::size_type>(reserve_size));

        std::ifstream fs(path.string().c_str());

        if (!fs)
            throw Exception::ItemNotOpened(MakeMessage(FMT_NOT_OPENED_FOR_READ, path), MakeAttributes(path));

        for (std::ifstream::char_type c; fs.get(c);)
            data.push_back(c);

        fs.close();
    }

    /*static*/void FileUtils::WriteBinaryFile(String const &filePathname, Binary const &data)
    {
        BOOST_STATIC_ASSERT(sizeof(Binary::value_type) == sizeof(std::ofstream::char_type));

        bfs::path path = filePathname;

        std::ofstream fs(path.string().c_str(), std::ios_base::out | std::ios_base::binary);

        if (!fs)
            throw Exception::ItemNotOpened(MakeMessage(FMT_NOT_OPENED_FOR_WRITE, path), MakeAttributes(path));

        fs.write(reinterpret_cast<char const *>(&data[0]), data.size());
        fs.close();
    }

    /*static*/void FileUtils::WriteStringFile(String const &filePathname, String const &data)
    {
        bfs::path path = filePathname;

        std::ofstream fs(path.string().c_str());

        if (!fs)
            throw Exception::ItemNotOpened(MakeMessage(FMT_NOT_OPENED_FOR_WRITE, path), MakeAttributes(path));

        for (String::const_iterator i = data.begin(), n = data.end(); i != n; ++i)
            fs.put(*i);

        fs.close();
    }

    /*static*/void FileUtils::AppendToBinaryFile(String const &filePathname, Binary const &data)
    {
        BOOST_STATIC_ASSERT(sizeof(Binary::value_type) == sizeof(std::ofstream::char_type));

        bfs::path path = filePathname;

        std::ofstream fs(path.string().c_str(), std::ios_base::app | std::ios_base::binary);

        if (!fs)
            throw Exception::ItemNotOpened(MakeMessage(FMT_NOT_OPENED_FOR_APPEND, path), MakeAttributes(path));

        fs.write(reinterpret_cast<char const *>(&data[0]), data.size());
        fs.close();
    }

    /*static*/void FileUtils::AppendToStringFile(String const &filePathname, String const &data)
    {
        bfs::path path = filePathname;

        std::ofstream fs(path.string().c_str(), std::ios_base::app);

        if (!fs)
            throw Exception::ItemNotOpened(MakeMessage(FMT_NOT_OPENED_FOR_APPEND, path), MakeAttributes(path));

        for (String::const_iterator i = data.begin(), n = data.end(); i != n; ++i)
            fs.put(*i);

        fs.close();
    }

    /*static*/void FileUtils::MakeDirectory(String const &directoryPathname)
    {
        bfs::path const directoryPath(directoryPathname);

        if (bfs::exists(directoryPath))
            throw Exception::ItemExists(MakeMessage(FMT_ALREADY_EXISTS, directoryPath), MakeAttributes(directoryPath));

        bfs::create_directory(directoryPath);
    }

    /*static*/void FileUtils::Remove(String const &pathname)
    {
        bfs::path const path(pathname);

        if (!bfs::exists(path))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, path), MakeAttributes(path));

        bfs::remove(path);
    }

    /*static*/ void FileUtils::RemoveAll(String const &directoryPathname)
    {
        bfs::path const dirPath(directoryPathname);

        if (!bfs::exists(dirPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, dirPath), MakeAttributes(dirPath));

        bfs::remove_all(dirPath);
    }

    /*static*/ void FileUtils::ClearFolder(String const &directoryPathname)
    {
        bfs::path const dirPath(directoryPathname);

        if (!bfs::exists(dirPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, dirPath), MakeAttributes(dirPath));

        std::vector<bfs::directory_entry> v; // To save the file names in a vector.
        std::copy(bfs::directory_iterator(dirPath), bfs::directory_iterator(), std::back_inserter(v));

        for (const auto& entry : v)
        {
            if (bfs::is_regular_file(entry.path()))
                bfs::remove(entry.path());
        }
    }

    /*static*/void FileUtils::FileCopy(String const &fromFilePathname, String const &intoDirectoryPathname, bool const writable)
    {
        bfs::path const fromFilePath(fromFilePathname);

        if (!bfs::exists(fromFilePath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, fromFilePath), MakeAttributes(fromFilePath));

        if (!bfs::is_regular_file(fromFilePathname))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_REGULAR, fromFilePath), MakeAttributes(fromFilePath));

        bfs::path const intoDirectoryPath(intoDirectoryPathname);

        if (!bfs::is_directory(intoDirectoryPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_REGULAR, intoDirectoryPath), MakeAttributes(intoDirectoryPath));

        bfs::path const toFilePath(intoDirectoryPath / fromFilePath.filename());

        if (bfs::exists(toFilePath))
            throw Exception::ItemNotFound(MakeMessage(FMT_ALREADY_EXISTS, toFilePath), MakeAttributes(toFilePath));

        bfs::copy_file(fromFilePath, toFilePath);

        if (writable)
            bfs::permissions(toFilePath, bfs::add_perms | bfs::owner_write | bfs::group_write | bfs::others_write);
    }

    /*static*/void FileUtils::FileDuplicate(String const &fromFilePathname, String const &intoDirectoryPathname)
    {
        bfs::path const fromFilePath(fromFilePathname);

        if (!bfs::exists(fromFilePath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, fromFilePath), MakeAttributes(fromFilePath));

        if (!bfs::is_regular_file(fromFilePathname))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_REGULAR, fromFilePath), MakeAttributes(fromFilePath));

        bfs::path const intoDirectoryPath(intoDirectoryPathname);

        if (!bfs::is_directory(intoDirectoryPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_REGULAR, intoDirectoryPath), MakeAttributes(intoDirectoryPath));

        bfs::path const toFilePath(intoDirectoryPath / fromFilePath.filename());

        if (bfs::exists(toFilePath))
            throw Exception::ItemNotFound(MakeMessage(FMT_ALREADY_EXISTS, toFilePath), MakeAttributes(toFilePath));

        Binary data;
        ReadBinaryFile(fromFilePath.string(), data);
        WriteBinaryFile(toFilePath.string(), data);
    }

    /*static*/void FileUtils::Rename(String const &fromPathname, String const &toPathname)
    {
        bfs::path const fromPath(fromPathname);

        if (!bfs::exists(fromPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, fromPath), MakeAttributes(fromPath));

        bfs::path const toPath(toPathname);

        if (bfs::exists(toPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_ALREADY_EXISTS, toPath), MakeAttributes(toPath));

        bfs::rename(fromPath, toPath);
    }

    /*static*/String FileUtils::GetTemporaryDirectoryPathname(String const &applicationSubPathname)
    {
        /*
        Hi Boon Kian,

            We need to remember to better understand why the 3070 Java application is changing the 
            environment variable(s) used by boost::filesystem to determine the path to a temporary directory.

            The change I put in means that we no longer exhibit the following behavior:
            ISO/IEC 9945:  The path supplied by the first environment variable found in
            the list TMPDIR, TMP, TEMP, TEMPDIR.  If none of these are found, "/tmp".
            What we have now is a hard-coded solution that works for the current version of Window.

            Long term this will be a maintenance and portability problem!

        Hi Rod,

            Do you have any idea about why the 3070 Java application is changing the environment (most likely TMP)
            from the windows default of:

            %USERPROFILE%\AppData\Local\Temp
            to:
            C:\Agilent_ICT\tmp

            The new directory has the following issues when trying to use the x1149 software from within the 3070 software:
            1.  [Problem] This directory is write protected.
            2.  [Potential Problem] This directory is shared by all users on the same system 
        */
#ifndef _WIN32_WINNT
        bfs::path path = bfs::temp_directory_path() / bfs::path(applicationSubPathname);
#else
        static String const name("USERPROFILE");
        static String const tail("\\AppData\\Local\\Temp");
        String pathname(GetEnvironmentValue(name));
        bfs::path path = bfs::path(pathname) / bfs::path(tail) / bfs::path(applicationSubPathname);
#endif
        return path.string();
    }

    /*static*/String FileUtils::GetTemporaryFilePathname(String const &extension, String const &applicationSubPathname)
    {
        String const PREFIX = SL("absa-");
        Int32  const TRIES  = 10;

        for (Int32 i = 0; i != TRIES; ++i)
        {
            bfs::path dirs = GetTemporaryDirectoryPathname(applicationSubPathname);
            bfs::path base = bfs::path(PREFIX + bfs::unique_path().string());
            bfs::path path = dirs / base;
            path.replace_extension(bfs::path(extension));

            if (!bfs::exists(path))
                return  path.string();
        }

        throw Exception::InternalError(SL("FileUtils::GetTemporaryFilePathname: A temporary file path could not be generated."));
    }

    /*static*/String FileUtils::GetPathInInstall(String const & relativePathname)
    {
        bfs::path module = GetInitialPathname();
        bfs::path const path = module.bfs::path::remove_filename() / bfs::path(relativePathname);
        return path.string();
    }

    /*static*/void FileUtils::GetPathsInDirectory(String const &directoryPathname, StringVector &paths)
    {
        StringVector().swap(paths);
        bfs::path const directoryPath = directoryPathname;

        if (!bfs::exists(directoryPath))
            throw Exception::ItemNotFound(MakeMessage(FMT_NOT_FOUND, directoryPath), MakeAttributes(directoryPath));

        if (!bfs::is_directory(directoryPath))
            throw Exception::InvalidArgument(MakeMessage(FMT_NOT_DIRECTORY, directoryPath), MakeAttributes(directoryPath));


        bfs::directory_iterator const b(directoryPath);
        bfs::directory_iterator const e;

        for (bfs::directory_iterator i = b; i != e; ++i)
        {
            bfs::directory_entry entry(*i);
            paths.push_back(entry.path().string());
        }
    }
}
}
