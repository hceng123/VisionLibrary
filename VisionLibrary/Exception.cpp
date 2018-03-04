//****************************************************************************
// Exception.cpp -- $Id$
//
// Purpose
//   Implements the exception framework.
//
// Indentation
//   Four characters. No tabs!
//
// Modifications
//   2011-09-25 (MM) Added an internal error thrower.
//   2011-02-09 (MM) Started using exception specifications; added new
//                   attribute key.
//   2010-12-03 (MM) Changed over to "std::runtime_error" as the base.
//   2010-08-29 (MM) Changed over to using Boost's format capabilities.
//   2010-08-10 (MM) Created.
//
// Copyright (c) 2010-2011 Agilent Technologies, Inc.  All rights reserved.
//****************************************************************************

#include "BaseDefs.h"
#include "Exception.h"

#define SL(s) s

namespace AOI
{
namespace Vision
{
namespace Exception
{
    namespace
    {
        String MakeDetails(String const &type, Int32 typeNumber, String const &message)
        {
            static String::value_type const fmt[] = SL("%s (%d) %s");
            return (boost::format(fmt) % type % typeNumber % message).str();
        }
    }

    /*static*/String const Runtime::attrType   (SL("type"));
    /*static*/String const Runtime::attrName   (SL("name"));
    /*static*/String const Runtime::attrDesc   (SL("desc"));
    /*static*/String const Runtime::attrPath   (SL("path"));
    /*static*/String const Runtime::attrSubtype(SL("subtype"));
    /*static*/String const Runtime::attrValue  (SL("value"));
    /*static*/String const Runtime::attrError  (SL("error"));

    Runtime::Runtime(String const &type, Int32 typeNumber, String const &message, Attributes const &attributes)
      : std::runtime_error(message),
        xType      (type),
        xTypeNumber(typeNumber),
        xMessage   (message),
        xDetails   (MakeDetails(type, typeNumber, message)),
        xAttributes(attributes),
        xChild     (0)
    {
        /* nothing */
    }

    Runtime::Runtime(String const &type, Int32 typeNumber, String const &message, Runtime const *child, Attributes const &attributes)
      : std::runtime_error(message),
        xType      (type),
        xTypeNumber(typeNumber),
        xMessage   (message),
        xDetails   (MakeDetails(type, typeNumber, message)),
        xAttributes(attributes),
        xChild     ((child) ? child->Clone() : 0)
    {
        /* nothing */
    }

    Runtime::Runtime(Runtime const &other)
      : std::runtime_error(other.xMessage),
        xType      (other.xType),
        xTypeNumber(other.xTypeNumber),
        xMessage   (other.xMessage),
        xDetails   (other.xDetails),
        xAttributes(other.xAttributes),
        xChild     ((other.xChild) ? other.xChild->Clone() : 0)
    {
        /* nothing */
    }

    Runtime::~Runtime() throw()
    {
        delete xChild;
    }

    Runtime &Runtime::operator=(Runtime const &other)
    {
        if (this != &other)
        {
            std::runtime_error::operator=(other);

            xType       = other.xType;
            xTypeNumber = other.xTypeNumber;
            xMessage    = other.xMessage;
            xAttributes = other.xAttributes;
            xChild      = (other.xChild) ? other.xChild->Clone() : 0;
        }

        return *this;
    }

    char const *Runtime::what() const throw()
    {
        // This is safe as long as the client doesn't subsequently
        // invoke the assigment operator (which would be a totally
        // weird thing to do with an exception in any case).
        return xDetails.c_str();
    }

    String const &Runtime::GetType() const
    {
        return xType;
    }

    Int32 Runtime::GetTypeNumber() const
    {
        return xTypeNumber;
    }

    String const &Runtime::GetMessage() const
    {
        return xMessage;
    }

    String const &Runtime::GetDetails() const
    {
        return xDetails;
    }

    Attributes const &Runtime::GetAttributes() const
    {
        return xAttributes;
    }

    void Runtime::SwapAttributes(Attributes &attributes) const
    {
        xAttributes.swap(attributes);
    }

    Runtime const *Runtime::Child() const
    {
        return xChild;
    }

    void ThrowInternalError(String const &message, String const &thrower, String const &fileName, Int32 lineNumber)
    {
        String const fmt = SL("%s [Source: %s; File: %s; Line %d]");
        String const src = (thrower.empty()) ? String(SL("Anonymous")) : thrower;
        throw Exception::InternalError((boost::format(fmt) % message % src % fileName % lineNumber).str());
    }
}
}
}
