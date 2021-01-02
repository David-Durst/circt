//===- Ops.h - SequenceLanguage MLIR Operations ----------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SequenceLanguage/SequenceLanguage.h"
#include "circt/Dialect/SequenceLanguage/Types.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace mlir;
using namespace circt;
using namespace circt::sequencelanguage;

static LogicalResult verifyMapTypesMatch(MapOp *op) {
    FunctionType f_t = op->f().getType().cast<FunctionType>();
    Type in_t = op->in().getType();
    if (f_t.getInput(0) != in_t) {
        return op->emitError("Map f input type 0 '")
            << f_t.getInput(0)
            << " doesn't match map input type"
            << in_t;
    }

    Type out_t = op->out().getType();
    if (f_t.getResult(0) != out_t) {
        return op->emitError("Map f output type '")
            << f_t.getResult(0)
            << " doesn't match map output type"
            << out_t;
    }

    return success();
}

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

