#ifndef _AOI_COMMON_BASETYPES_H
#define _AOI_COMMON_BASETYPES_H

#include <string>
#include <vector>

#define SL(s) s

namespace AOI
{
    typedef unsigned __int8  Byte;
    typedef __int16          Int16;
	typedef unsigned __int16 UInt16;
    typedef __int32          Int32;
    typedef __int64          Int64;
    typedef unsigned int     UInt32;
    typedef std::string      String;

    typedef std::vector<Byte>   Binary;
    typedef std::vector<Byte>   ByteVector;
    typedef std::vector<Int32>  Int32Vector;
    typedef std::vector<Int64>  Int64Vector;
    typedef std::vector<String> StringVector;

    typedef std::pair<Int32,  Int32>  Int32Pair;
    typedef std::pair<Int64,  Int64>  Int64Pair;
    typedef std::pair<String, String> StringPair;
    typedef std::pair<Int32,  String> Int32String;
    typedef std::pair<Int64,  String> Int64String;

    typedef std::vector<Int32Pair>   Int32PairVector;
    typedef std::vector<Int64Pair>   Int64PairVector;
    typedef std::vector<StringPair>  StringPairVector;
    typedef std::vector<Int32String> Int32StringVector;
    typedef std::vector<Int64String> Int64StringVector;

    // This is taken from item 6 (pp. 39) of "Effective C++, 3rd Ed." by
    // Scott Meyers. Declaring it as a private base class does the trick.
    // If you need a virtual destructor, use "boost::noncopyable" instead.
    class Uncopyable
    {
        Uncopyable(Uncopyable const &);
        Uncopyable &operator=(Uncopyable const &);

    protected:
        Uncopyable() {}
        ~Uncopyable() {} /* Intentionally not virtual. */
    };
}

#endif