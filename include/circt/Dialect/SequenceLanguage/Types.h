
//===- Types.h - types for the RTL dialect ----------------------*- C++ -*-===//
//
// Types for the Sequence Language dialect are mostly in tablegen. This file
// should contain C++ types used in MLIR type parameters (including import the
// generated SequenceLanguageTypes.h.inc from the tablegen file).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SEQUENCELANGUAGE_TYPES_H
#define CIRCT_DIALECT_SEQUENCELANGUAGE_TYPES_H

#include "mlir/IR/Types.h"

#define GET_SEQ_TYPEDEF_CLASSES
#include "circt/Dialect/SequenceLanguage/SequenceLanguageTypes.h.inc"

#endif // CIRCT_DIALECT_SEQUENCELANGUAGE_TYPES_H
