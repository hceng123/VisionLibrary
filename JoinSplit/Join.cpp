/* ************************************************************************
FILE    : JOIN.C
PURPOSE : Takes a list of files with the same name but different extensions
        and joins them all into a single file with a new extension (.WB).
        The file has a header which contains details on the files contained
        there in.
HISTORY:
    21 Dec 99   JPT @1
    Many functions changed to have their arguments as const so that
    any modification of the arguments accidentally can be detected by
    the compiler.
    27 Jun 00   JPT @2
    Updated to support GZIP compression.
    16 Jan 04   Wang xiaoqing @3
    support to join a whole directory
    10 Nov 2004	-Zhan Zhuoqun @4
    use another communication channel with PR.
    11 Nov 05 -- wenhong @5
    - Display error message once can't zip all files.
	14 Dec 05 -- wenhong @6
	- Improve GZIP utility. In order to be compatible with Twin Eagle which
	Gzip is still old version, for bond programs with less than 128 files,
	base on version 1.0 to zip. For bond programs with more than 128 files, 
	base on version 2.0 to zip.	
    06 October 2007       Guo Jianhua @7
    global var replace, use 'STDERR' replace 'stderr'in order to fix bug for WB_fprintf().
*/
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>
#include <string.h>
#include <dos.h>
#include <errno.h>
#include <assert.h>
#include <fcntl.h>
#include <io.h>
#include <exception>

#include "zlib.h"
#include "DataStruct.h"

#ifndef GZ_SUFFIX
#  define GZ_SUFFIX ".gz"
#endif
#define SUFFIX_LEN (sizeof(GZ_SUFFIX)-1)

#define BUFLEN      32767
#define MAX_NAME_LEN 1024

static stJAndSHeader* pstrFileNameArr=NULL;	//@4
void error(const char *msg);
extern short file_compress(const char *szDir, const char *szFileName,
    const char *szSrcExt, const char *szZipExt, const char *mode);
//extern FILE *fileopen( const char *filename, const char *mode );
static int writeHeader ( stJAndSHeader *ptrHeader, FILE *fileHdl);
static int concatFiles ( const char *sourceDir, stJAndSHeader *ptrHeader,
    FILE *fileHdl);


void Append_OneItem_FileNameArr(char *filename, short size)
{
	if(pstrFileNameArr->numFilesJoined<MAX_FILES_JOINED){
		strcpy(pstrFileNameArr->fileName[pstrFileNameArr->numFilesJoined],filename);
		pstrFileNameArr->fileLen[pstrFileNameArr->numFilesJoined]=size;
		pstrFileNameArr->numFilesJoined++;
	}
}

int joinFiles_Ext1 (const char *sourceDir, const char *sourceName,const char *extName)
{
	stJAndSHeader	*joinHdr = NULL;
    FILE	*fileHdl = NULL;
    char	destName[_MAX_DIR],tempName[_MAX_DIR];
    char 	outmode[] = "wb6 ";
    short	nErr;

    // Get memory for the join header structure
    joinHdr=pstrFileNameArr;
    if(joinHdr->numFilesJoined==0)
    {
    	return 1;
    }
    
	errno = EZERO;
	// Generate the destination name
	if (strlen(sourceDir))
		sprintf (destName, "%s\\%s%s", sourceDir, sourceName, extName);
	else
		sprintf (destName, ".\\%s%s", sourceName, extName);

	// Open the join file...
	if ((fileHdl = fopen (destName, "wb")) != NULL)
	{
		// ... and join in all the files.
		writeHeader ( joinHdr, fileHdl);
		concatFiles ( sourceDir, joinHdr, fileHdl);
		fclose (fileHdl);

		// Rename file as a temporary file
		if (strlen(sourceDir))
			sprintf (tempName, "%s\\%s%s", sourceDir, sourceName, ".CAT");
		else
			sprintf (tempName, ".\\%s%s", sourceName, ".CAT");

		rename (destName, tempName);

		// Compress the file @2              
		if ((nErr = file_compress (sourceDir, sourceName, ".CAT", extName,outmode)) != 0)
		{
			String msg = String("Failed to compress ") + sourceName;
            throw std::exception ( msg.c_str() );
			return nErr;
		}
	}
    return errno;
}

/* ===========================================================================
 * Display error message and exit @2
 */
static void error( const char *msg )
{
    throw std::exception ( msg );
}


/* ===========================================================================
 * Compress input to output then close both files. @2
 */

short gz_compress(FILE *in, gzFile out)
{
    char
        *buf = NULL;
    size_t len;
    int err;

    if ((buf = (char*)malloc (BUFLEN)) == NULL)
    {
        throw std::exception ("Insufficient Memory for read buffer" );
        return 3;
    }

    for (;;) {
        len = fread(buf, 1, BUFLEN, in);
        if (ferror(in)) {
            perror("fread");
            free (buf);
            return 3;
        }

        if (len == 0)
            break;

        if (gzwrite(out, buf, (unsigned)len) != len)
        {
            error(gzerror(out, &err));
            free (buf);
            return 3;
        }
    }
    fclose(in);

    if (gzclose(out) != Z_OK)
    {
        error("failed gzclose");
        free (buf);
        return 3;
    }

    free (buf);
    return 0;
}

/* ===========================================================================
 * Compress the given file: create a corresponding .gz file and remove the
 * original. @2
 */
short file_compress(const char *szDir, const char *szFileName,
    const char *szSrcExt, const char *szZipExt, const char *mode)
{
    char
        szSourceFile[MAX_NAME_LEN],
        szGzipFile[MAX_NAME_LEN];
    FILE  *in;
    gzFile out;

    if (strlen(szDir))
        strcpy (szSourceFile, szDir);
    else
        strcpy (szSourceFile, ".");
    strcat (szSourceFile, "\\");
    strcat (szSourceFile, szFileName);
    strcpy (szGzipFile,   szSourceFile);
    strcat (szSourceFile, szSrcExt);
    strcat (szGzipFile,   szZipExt);

    in = fopen(szSourceFile, "rb");
    if (in == NULL) {
        perror(szSourceFile);
        return 1;
    }
    out = gzopen(szGzipFile, mode);
    if (out == NULL) {
        fclose(in);	//@4
        String msg = String ( "can't gzopen " ) + szGzipFile;
        throw std::exception( msg.c_str() );
        return 1;
    }
    gz_compress(in, out);

    _unlink(szSourceFile);
    fclose(in);	//@4
    return 0;
}

/* ========================================================================
PURPOSE : Searches through the pathName directory for files matching the
        wildcard baseName.* and fills in all the data to the ptrHeader
        structure.
RETURNS : Number of files encoded into the header.
HISTORY: @1
*/
static int genHeader (const char *pathName, const char *baseName,
    const char *extName, stJAndSHeader *ptrHeader, char dir_flag) //@3
{
    char
        wildcardName[_MAX_DIR],
        joinFileName[_MAX_DIR];
    //struct find_t
    //    fileinfo;
    struct _finddata_t
        fileinfo;
    unsigned
        rc = 0;        /* return code */

    // No files found yet
    (ptrHeader->numFilesJoined) = 0;

    // Generate the wild card name to match against
    if (dir_flag == 1) { //@3
        if (strlen(pathName))
            sprintf (wildcardName, "%s\\*.*", pathName);
        else
            sprintf (wildcardName, ".\\*.*");
    }
    else {
        if (strlen(pathName))
            sprintf (wildcardName, "%s\\%s.*", pathName, baseName);
        else
            sprintf (wildcardName, ".\\%s.*", baseName);
    }

    // Generate name of the join file
    _snprintf (joinFileName, 200, "%s%s", baseName, extName);

    // Search for all files matching the wildcard
    //rc = _dos_findfirst (wildcardName, _A_NORMAL, &fileinfo );
    auto handle = _findfirst(wildcardName, &fileinfo);

	if (handle != -1)
	{
        // Get the data for each file found upto max that can be stored
		//while ((rc == 0) &&
		//		(ptrHeader->numFilesJoined < MAX_FILES_JOINED))
		do
        {
            // Don't include files with the same extension as the join file
            if (_stricmp(joinFileName, fileinfo.name) != 0 && fileinfo.size > 0 )
            {
                // Copy the file information to the header
                _snprintf (ptrHeader->fileName[ptrHeader->numFilesJoined],15,
                        fileinfo.name);
                ptrHeader->fileLen[ptrHeader->numFilesJoined] =
                        fileinfo.size;
                (ptrHeader->numFilesJoined)++;
            }
    
            // Look for the next file in the directory
			//rc = _dos_findnext( &fileinfo );
			rc = _findnext(handle, &fileinfo);
		} while ((rc == 0) && (ptrHeader->numFilesJoined < MAX_FILES_JOINED));
		_findclose(handle);
    }
    
    //@5
    if ((rc == 0)&&(ptrHeader->numFilesJoined == MAX_FILES_JOINED))
    {
        // Don't include files with the same extension as the join file
        if (_stricmp(joinFileName, fileinfo.name) != 0)  
        {
        	throw std::exception ("Too many files joined!!!" );
        }
    }

    return ptrHeader->numFilesJoined;
}

#ifdef DEBUG_MODE
/* ========================================================================
PURPOSE:
*/
static printHeader (FILE *fHandle, stJAndSHeader *ptrHeader)
{
    int i;
    long
        totalSize = 0;

    for (i=0; i<ptrHeader->numFilesJoined; i++)
    {
        WB_fprintf( WB_STDERR, "[%3d] %13s %5ld\n", i,
            ptrHeader->fileName[i], ptrHeader->fileLen[i]);
        totalSize += ptrHeader->fileLen[i];
    }
    WB_fprintf( WB_STDERR, "Total File size = %ld\n", totalSize);
}
#endif

/* ========================================================================
PURPOSE :
*/
static int writeHeader ( stJAndSHeader *ptrHeader, FILE *fileHdl)
{
    char bufID[MAX_FILE_ID] = { 0 };

    // Copy in the version number
    strcpy ( bufID, JANDS_ID );

    // Ensure don't go over the buffer limit
    assert ( strlen ( bufID ) < MAX_FILE_ID );

    fwrite ( bufID, MAX_FILE_ID, 1, fileHdl );
    fwrite ( ptrHeader, sizeof (stJAndSHeader), 1, fileHdl );
    return errno;
}

/* ========================================================================
PURPOSE :
HISTORY: @1
*/
static long writeFile (const char *sourceDir, char *fileName, long fileLen, FILE *fileHdl)
{
    char
        *ptrBuf;
    char
        sourceName[_MAX_DIR];
    FILE
        *sourceHdl = NULL;

    // Get memory for the file storage
    ptrBuf = (char*)malloc(fileLen); //@swq for kw, move out from condition.
    if (ptrBuf != NULL)
    {
        if (strlen(sourceDir))
            sprintf (sourceName, "%s\\%s", sourceDir, fileName);
        else
            sprintf (sourceName, ".\\%s", fileName);

        // Copy the source file to memory
        if ( (sourceHdl = fopen ( sourceName, "rb" )) != NULL )
        {
            if (fread(ptrBuf, fileLen, 1, sourceHdl) != 1)
            {
				throw std::exception ("Read error");
            }
			else
			{
				// Append the file contents to the join file
				if (fwrite(ptrBuf, fileLen, 1, fileHdl) != 1)
				{
					throw std::exception ("Write error" );
				}

			}
            fclose (sourceHdl);
        }
        free(ptrBuf);
        ptrBuf = NULL;
    }
    else
		throw std::exception ( "Insufficient memory" );

    return errno;
}

/* ========================================================================
PURPOSE :
HISTORY: @1
*/
static int concatFiles ( const char *sourceDir, stJAndSHeader *ptrHeader, FILE *fileHdl)
{
    // For each file in the header concat to the new join file
    for (int i = 0; i < ptrHeader->numFilesJoined; ++ i)
    {
        // Concat this file to the end of the current join file
        if ( writeFile (sourceDir, ptrHeader->fileName[i], ptrHeader->fileLen[i], fileHdl) != 0 )
        {
            String msg = String ( "Failed to join in file %s" ) + ptrHeader->fileName[i];
			throw std::exception ( msg.c_str() );
        }
    }

    return errno;
}

/* ========================================================================
PURPOSE :
*/
int joinFiles (const char *sourceDir, const char *sourceName, const char *extName)
{
    stJAndSHeader
        *joinHdr = NULL;
    FILE
        *fileHdl = NULL;
    char
        destName[_MAX_DIR],
        tempName[_MAX_DIR];
    char outmode[] = "wb6 ";
    short
        nErr;

    // Clear any outstanding errors
    errno = EZERO;

    // Get memory for the join header structure
    if ((joinHdr = (stJAndSHeader*)malloc(sizeof(stJAndSHeader))) != NULL)
    {
        // Populate the header with all the match files
        if (genHeader (sourceDir, sourceName, extName, joinHdr, 0)) //@3 files
        {
            // Clear any outstanding errors
            errno = EZERO;
            // Generate the destination name
            if (strlen(sourceDir))
                sprintf (destName, "%s\\%s%s", sourceDir, sourceName, extName);
            else
                sprintf (destName, ".\\%s%s", sourceName, extName);

            // Open the join file...
            if ((fileHdl = fopen (destName, "wb")) != NULL)
            {
                // ... and join in all the files.
                writeHeader ( joinHdr, fileHdl);
                concatFiles ( sourceDir, joinHdr, fileHdl);
                fclose (fileHdl);

                // Rename file as a temporary file
                if (strlen(sourceDir))
                    sprintf (tempName, "%s\\%s%s", sourceDir, sourceName, ".CAT");
                else
                    sprintf (tempName, ".\\%s%s", sourceName, ".CAT");
                rename (destName, tempName);

                // Compress the file @2
                if ((nErr = file_compress (sourceDir, sourceName, ".CAT", extName, outmode)) != 0)
                {                    
                    free(joinHdr);
                    String msg = String("Failed to compress ") + sourceName;
                    throw std::exception ( msg.c_str() );
                    return nErr;
                }
            }
        }
        free(joinHdr);
        joinHdr = NULL;
    }
    else
        throw std::exception( "Insufficient memory" );

    // For debugging only
    #ifdef DEBUG_MODE
    printHeader (stdout,joinHdr);
    #endif

    return errno;
}

int joinDir(const char *sourceDir, const char *extName)
{
    stJAndSHeader *joinHdr = NULL;
    FILE *fileHdl = NULL;
    char
        destName[_MAX_DIR],
        tempName[_MAX_DIR];
    char outmode[] = "wb6 ";
    short  nErr;

    // Clear any outstanding errors
    errno = EZERO;

    // Get memory for the join header structure
    if ((joinHdr = (stJAndSHeader*)malloc(sizeof(stJAndSHeader))) != NULL)
    {
        // Populate the header with all the match files
        if (genHeader (sourceDir, "", extName, joinHdr, 1)) //@3 directory
        {
            // Clear any outstanding errors
            errno = EZERO;

            // Generate the destination name
            sprintf (destName, "%s%s", sourceDir, extName);

            // Open the join file...
            if ((fileHdl = fopen (destName, "wb")) != NULL)
            {
                // ... and join in all the files.
                writeHeader ( joinHdr, fileHdl);
                concatFiles ( sourceDir, joinHdr, fileHdl);
                fclose (fileHdl);

                // Rename file as a temporary file
                sprintf (tempName, "%s%s", sourceDir, ".CAT");

                rename (destName, tempName);

                String strDir(sourceDir);
                size_t pos = strDir.find_last_of ( '\\' );
                if ( String::npos == pos )
                    pos = strDir.find_last_of ( '/' );
                String strFolderName = strDir.substr ( pos + 1 );
                String strParentDir = strDir.substr(0, pos);

                // Compress the file @2
                if ( ( nErr = file_compress ( strParentDir.c_str(), strFolderName.c_str(), ".CAT", extName, outmode ) ) != 0 )
                {                    
                    free(joinHdr);
                    String msg = String("Failed to compress ") + tempName;
                    throw std::exception ( msg.c_str() );
                    return nErr;
                }
            }
        }
        free(joinHdr);
        joinHdr = NULL;
    }
    else
        throw std::exception("Insufficient memory");

    // For debugging only
    #ifdef DEBUG_MODE
    printHeader (stdout,joinHdr);
    #endif

    return errno;
}

#ifdef TEST_MODE
/* ========================================================================
PURPOSE : For functional test verification only.
*/
void main (int argc, char **argv)
{
    char
        extName[] = {".WB"},
        szPath[128],
        szDrive[128],
        szDir[128],
        szFname[128],
        szExt[128];
    int
        error = 0;
    char outmode[20];

    strcpy(outmode, "wb6 ");

    // Handler user input
    if (argc < 2)
        WB_fprintf( WB_STDERR, "%s <BASENAME>\n", argv[0]);
    else
    {
        printf ("Generating File %s%s\n", argv[1], extName);
        strcpy (szPath, argv[1]);
        _splitpath (szPath, szDrive, szDir, szFname, szExt);
        if ((error =joinDir (szDir,szFname, extName)) != 0)
        {
            WB_fprintf( WB_STDERR, "Error %d\n", error);
        }
    }
}
#endif
