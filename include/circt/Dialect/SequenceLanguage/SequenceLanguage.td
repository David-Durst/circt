//===- SequenceLanguage.td - SequenceLanguage Definitions ------------*- tablegen -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifdef SEQUENCELANGUAGE_OPS
#else
#define SEQUENCELANGUAGE_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SequenceLanguage_Dialect : Dialect {
  let name = "sequencelanguage";
  let cppNamespace = "::circt::sequencelanguage";
}

def AbsOp : Op<SequenceLanguage_Dialect, "abs", [NoSideEffect]> {
  let summary = "absolute value operation";
  let description = [{
    The "sequencelanguage.abs" operation represents an absolute value on an int atom.
  }];

  let arguments = (ins AnySignlessInteger:$in);
  let results = (outs AnySignlessInteger:$out);

  //let verifier = [{ this-> }];
}

def PipelineOp : Op<SequenceLanguage_Dialect, "pipeline", [NoSideEffect]> {
  let summary = "pipeline operation";
  let description = [{
    The "staticlogic.pipeline" operation represents a statically scheduled 
    pipeline stucture which contains several MLIR blocks. Each MLIR block is 
    corresponding to a pipeline stage.
  }];

  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region AnyRegion: $body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilderDAG<(ins "ValueRange":$operands, "ValueRange":$results), [{
    SmallVector<Type, 4> argTypes;
    for (auto value : operands)
      argTypes.push_back(value.getType());

    SmallVector<Type, 4> resultTypes;
    for (auto value : results)
      resultTypes.push_back(value.getType());

    Region *bodyRegion = odsState.addRegion();
    Block *body = new Block();
    bodyRegion->push_back(body);
    body->addArguments(argTypes);

    odsState.addOperands(operands);
    odsState.addTypes(resultTypes);
  }]>];
}

def ReturnOp : Op<SequenceLanguage_Dialect, "return", [Terminator]> {
  let summary = "SequenceLanguage dialect return.";
  let description = [{
    The "sequencelanguage.return" operation represents a terminator of a statically 
    scheduled module, which is similar to a standard return operation.
  }];

  let arguments = (ins Variadic<AnyType>: $operands);
}

#endif // SEQUENCELANGUAGE_OPS
