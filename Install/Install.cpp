// Install.cpp : Defines the entry point for the console application.
//

#include "tchar.h"
#include "../VisionLibrary/VisionAPI.h"
#include "boost/filesystem.hpp"
#include <iostream>

namespace bfs = boost::filesystem;

using namespace AOI;

int _tmain(int argc, _TCHAR* argv[])
{
    try {
        Vision::PR_VERSION_INFO stVersion;
        Vision::PR_GetVersion(&stVersion);
        std::string strDestFolder = std::string("VisionLibrary_") + stVersion.chArrVersion;
        if (bfs::exists(strDestFolder))
            bfs::remove_all(strDestFolder);

        bfs::create_directory(strDestFolder);
        std::string strInclude = strDestFolder + "/include";
        if (!bfs::exists(strInclude))
            bfs::create_directory(strInclude);

        std::string strX64 = strDestFolder + "/x64";
        if (!bfs::exists(strX64))
            bfs::create_directory(strX64);

        std::string strX64Debug = strX64 + "/Debug";
        if (!bfs::exists(strX64Debug))
            bfs::create_directory(strX64Debug);

        std::string strX64Release = strX64 + "/Release";
        if (!bfs::exists(strX64Release))
            bfs::create_directory(strX64Release);

        bfs::copy_file("../VisionLibrary/VisionStruct.h", strInclude + "/VisionStruct.h");
        bfs::copy_file("../VisionLibrary/VisionType.h", strInclude + "/VisionType.h");
        bfs::copy_file("../VisionLibrary/VisionStatus.h", strInclude + "/VisionStatus.h");
        bfs::copy_file("../VisionLibrary/VisionAPI.h", strInclude + "/VisionAPI.h");
        bfs::copy_file("../VisionLibrary/BaseType.h", strInclude + "/BaseType.h");
        bfs::copy_file("../VisionLibrary/VisionHeader.h", strInclude + "/VisionHeader.h");

        bfs::copy_file("../Obj/VisionLibrary/x64_Debug/bin/VisionLibrary.dll", strX64Debug + "/VisionLibrary.dll");
        bfs::copy_file("../Obj/VisionLibrary/x64_Debug/bin/VisionLibrary.lib", strX64Debug + "/VisionLibrary.lib");
        bfs::copy_file("../Obj/VisionLibrary/x64_Debug/bin/VisionLibrary.pdb", strX64Debug + "/VisionLibrary.pdb");

        bfs::copy_file("../Obj/VisionLibrary/x64_Release/bin/VisionLibrary.dll", strX64Release + "/VisionLibrary.dll");
        bfs::copy_file("../Obj/VisionLibrary/x64_Release/bin/VisionLibrary.lib", strX64Release + "/VisionLibrary.lib");
        bfs::copy_file("../Obj/VisionLibrary/x64_Release/bin/VisionLibrary.pdb", strX64Release + "/VisionLibrary.pdb");
    }
    catch (const std::exception &e) {
        std::cout << "Exception encountered. " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

