#ifndef AGILENT_BBOX_COMMON_BASEDEFS_H
#define AGILENT_BBOX_COMMON_BASEDEFS_H
//****************************************************************************
// BaseDefs.h -- $Id$
//
// Purpose
//   Provides a base set of frequently-used declarations. This is a good
//   choice as the precompiled header file.
//
// Indentation
//   Four characters. No tabs!
//
// Modifications
//   2014-03-13 (MM) Added a file-utils include.
//   2014-02-10 (MM) Added two includes and a boost list size override.
//   2011-10-06 (MM) Added more includes.
//   2010-12-06 (MM) Under construction.
//   2010-08-09 (MM) Created.
//
// Copyright (c) 2010-2011, 2014 Agilent Technologies, Inc.  All rights reserved.
//****************************************************************************

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cctype>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#define  BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#define  BOOST_MPL_LIMIT_LIST_SIZE 30
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/utility.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/make_shared.hpp>
#include <boost/variant.hpp>
#include "BaseType.h"
#include "Exception.h"
#include "FileUtils.h"

#endif//AGILENT_BBOX_COMMON_BASEDEFS_H
