//===- SequenceLanguage.h - SequenceLanguage Definitions ------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SEQUENCELANGUAGE_OPS_H_
#define CIRCT_SEQUENCELANGUAGE_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace sequencelanguage {

using namespace mlir;

class SequenceLanguageDialect : public Dialect {
public:
  SequenceLanguageDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "sequencelanguage"; }
};

} // namespace sequencelanguage 
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/SequenceLanguage/SequenceLanguage.h.inc"

#endif // CIRCT_SEQUENCELANGUAGE_OPS_H_
