//****************************************************************************
// boost_main.cpp -- $Id$
//
// Purpose
//   Used primarily to aid in building boost libraries.
//
// Tabstops
//   No tabs!
//
// Modifications
//   2010-09-19 (MM) Created.
//
// Copyright (c) 2010 Agilent Technologies, Inc.  All rights reserved.
//****************************************************************************

#include "boost/shared_ptr.hpp"
#include "boost/static_assert.hpp"
#include "boost/utility.hpp"

int main()
{
    boost::shared_ptr<int> i(new int);
    *i = 0;
    return *i;
}
