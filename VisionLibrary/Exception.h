#ifndef AGILENT_BBOX_COMMON_EXCEPTION_H
#define AGILENT_BBOX_COMMON_EXCEPTION_H
//****************************************************************************
// Exception.h -- $Id$
//
// Purpose
//   Provides commonly-used exceptions as well as Exception::Runtime which
//   is used as the base class for exceptions.
//
// Indentation
//   Four characters. No tabs!
//
// Modifications
//   2015-01-15 (MM) Added the ProgrammingError exception class.
//   2014-03-05 (MM) Added the InvalidSyntax exception class.
//   2013-06-25 (MM) Added the eft-error exception.
//   2013-05-02 (MM) Changed NotPermitted to NotLicensed to be clearer.
//   2012-07-03 (MM) Added the action-required exception.
//   2012-03-23 (MM) Added three new item-not exceptions.
//   2011-09-25 (MM) Added an internal error thrower.
//   2011-08-02 (MM) Added InvalidSyntax.
//   2011-02-09 (MM) Started using exception specifications; added new
//                   attribute key; added TestheadError exception.
//   2010-12-03 (MM) Changed over to "std::runtime_error" as the base.
//   2010-08-10 (MM) Created.
//
// Copyright (c) 2010-2015 Keysight Technologies, Inc.  All rights reserved.
//****************************************************************************

#include "BaseDefs.h"

namespace AOI
{
namespace Vision
{
namespace Exception
{
    namespace Type
    {
        enum
        {
            // Because the exception values are negatives, they must
            // be individually set to avoid the enum incrementing in
            // the wrong direction.
            None                =  0,
            InternalError       = -1,
            NotImplemented      = -2,
            NotLicensed         = -3,

            ActionRequired      = -7,

            InvalidArgument     = -10,
            InvalidDatabase     = -11,
            InvalidConversion   = -12,
            InvalidObject       = -13,
            InvalidOperation    = -14,
            InvalidProperty     = -15,
            InvalidRange        = -16,
            InvalidSyntax       = -17,

            ItemExists          = -20,
            ItemNotCreated      = -21,
            ItemNotDeleted      = -22,
            ItemNotFound        = -23,
            ItemNotOpened       = -24,
            ItemNotReassociated = -25,
            ItemNotRenamed      = -26,
            ItemReadOnly        = -27,

            EftError            = -30,
            TestheadError       = -31,
            ProgrammingError    = -32,
        };
    }

    // The Exception::Attributes class provides support for name-value
    // attribution of exception. Examples include:
    //
    //   Attribute("type", "file")
    //   Attribute("name", "file.txt")
    //
    typedef std::pair<String, String> Attribute;
    typedef std::vector<Attribute>    Attributes;

    struct AttributeEqual: public std::binary_function<Attribute, Attribute, bool>
    {
        bool operator()(Attribute const &a1, Attribute const &a2) const
        {
            return a1.first == a2.first && a1.second == a2.second;
        }
    };

    // The Exception::Runtime class acts as the base class for all
    // exception types.
    class Runtime: public std::runtime_error
    {
        String   xType;
        Int32    xTypeNumber;
        String   xMessage;
        String   xDetails;
        Runtime *xChild;

        mutable Attributes xAttributes;

    protected:
        Runtime(String const &type, Int32 typeNumber, String const &message, Attributes const &attributes=Attributes());
        Runtime(String const &type, Int32 typeNumber, String const &message, Runtime const *child, Attributes const &attributes=Attributes());
        Runtime(Runtime const &);
        Runtime &operator=(Runtime const &);
       ~Runtime() throw();

    public:
        static String const attrType;    // "type"
        static String const attrName;    // "name"
        static String const attrDesc;    // "desc"
        static String const attrPath;    // "path"
        static String const attrSubtype; // "subtype"
        static String const attrValue;   // "value"
        static String const attrError;   // "error"

        virtual char const *what() const throw();

        virtual String const &GetType()       const;
        virtual Int32         GetTypeNumber() const;

        // The message is what was passed in when the exception was
        // thrown, while the details include the exception type and
        // number along with the message.
        virtual String const &GetMessage() const;
        virtual String const &GetDetails() const;

        virtual Attributes const &GetAttributes() const;

        // This not-really-const method provides a mechanism that
        // enables a client to acquire (aka steal) an exception's
        // attributes in a memory allocation-free manner.
        virtual void SwapAttributes(Attributes &) const;

        virtual Runtime const *Child() const;
        virtual Runtime       *Clone() const = 0;
    };

#define DECLARE_EXCEPTION(T) \
    class T: public Runtime \
    { \
    public: \
        explicit T(String const &message, Attributes const &attributes=Attributes()): Runtime(String(#T), Type::T, message, attributes) {} \
        T(String const &message, Runtime const *child, Attributes const &attributes=Attributes()): Runtime(String(#T), Type::T, message, child, attributes) {} \
        T(T const &t): Runtime(t) {} \
        T &operator=(T const &t) { Runtime::operator=(t); return *this; } \
        T *Clone() const { return new T(*this); } \
    } /* intentionally no semicolon */

    DECLARE_EXCEPTION(InternalError);
    DECLARE_EXCEPTION(NotImplemented);
    DECLARE_EXCEPTION(NotLicensed);
    DECLARE_EXCEPTION(ActionRequired);

    DECLARE_EXCEPTION(InvalidArgument);
    DECLARE_EXCEPTION(InvalidDatabase);
    DECLARE_EXCEPTION(InvalidConversion);
    DECLARE_EXCEPTION(InvalidObject);
    DECLARE_EXCEPTION(InvalidOperation);
    DECLARE_EXCEPTION(InvalidProperty);
    DECLARE_EXCEPTION(InvalidRange);
    DECLARE_EXCEPTION(InvalidSyntax);

    DECLARE_EXCEPTION(ItemExists);
    DECLARE_EXCEPTION(ItemNotCreated);
    DECLARE_EXCEPTION(ItemNotDeleted);
    DECLARE_EXCEPTION(ItemNotFound);
    DECLARE_EXCEPTION(ItemNotOpened);
    DECLARE_EXCEPTION(ItemNotReassociated);
    DECLARE_EXCEPTION(ItemNotRenamed);
    DECLARE_EXCEPTION(ItemReadOnly);

    DECLARE_EXCEPTION(EftError);
    DECLARE_EXCEPTION(TestheadError);
    DECLARE_EXCEPTION(ProgrammingError);
#undef  DECLARE_EXCEPTION

    // Example:
    //   ThrowInternalError(SL("Blah blah blah."), SL("SomeClass::SomeMethod"), __FILE__, __LINE__);
    void ThrowInternalError(String const &message, String const &thrower, String const &fileName, Int32 lineNumber);
}
}
}
#endif//AGILENT_BBOX_COMMON_EXCEPTION_H
