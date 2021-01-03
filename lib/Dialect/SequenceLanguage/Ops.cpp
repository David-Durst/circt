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
    SeqType in_t = op->in().getType().cast<SeqType>();
    if (f_t.getInput(0) != in_t.getT()) {
        return op->emitError("Map f input type ")
            << f_t.getInput(0)
            << " doesn't match element of input seq type"
            << in_t;
    }

    SeqType out_t = op->out().getType().cast<SeqType>();
    if (f_t.getResult(0) != out_t.getT()) {
        return op->emitError("Map f output type ")
            << f_t.getResult(0)
            << " doesn't match element of output seq type"
            << out_t;
    }

    if (in_t.getN() != out_t.getN()) {
        return op->emitError("Map input length ")
            << out_t.getN()
            << " doesn't match output length "
            << out_t.getN();
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

