//===- Ops.h - SequenceLanguage MLIR Operations ----------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SequenceLanguage/SequenceLanguage.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace circt;
using namespace circt::sequencelanguage;

static LogicalResult verifyMapTypesMatch2(MapOp *op) {
    FunctionType f_t = op->f().getType().cast<FunctionType>();
    Type in_t = op->in().getType();
    if (f_t.getInput(0) != in_t) {
        return op->emitError("Map f input type 0 '")
            << f_t.getInput(0)
            << " doesn't match map input type"
            << in_t;
    }

    return success();
}
