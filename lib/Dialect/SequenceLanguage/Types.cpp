//===- SequenceLanguageTypes.cpp - SequenceLanguage types code defs ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation logic for SequenceLanguage data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SequenceLanguage/SequenceLanguage.h"
#include "circt/Dialect/SequenceLanguage/Types.h"

using namespace mlir;
using namespace circt::sequencelanguage;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/SequenceLanguage/SequenceLanguageTypes.cpp.inc"
