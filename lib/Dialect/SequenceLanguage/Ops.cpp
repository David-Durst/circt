//===- Ops.h - SequenceLanguage MLIR Operations ----------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SequenceLanguage/SequenceLanguage.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::staticlogic;

#define GET_OP_CLASSES
#include "circt/Dialect/SequenceLanguage/SequenceLanguage.cpp.inc"

SequenceLanguageDialect::SequenceLanguageDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
    ::mlir::TypeID::get<SequenceLanguageDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SequenceLanguage/SequenceLanguage.cpp.inc"
      >();
}
