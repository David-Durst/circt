//===- Types.td - Sequence Language data type definitions --*- tablegen -*-===//
//
// Basic data types for the RTL dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class SeqLangType<string name> : TypeDef<SequenceLanguage_Dialect, name> { }

def SeqType : SeqLangType<"Seq"> {
  let summary = "A Sequence Language sequence of fixed length";
  let description = [{
    Sequence Language sequences describe the number of elements but not the
    layout in space or time.
  }];

  let mnemonic = "seq";
  let parameters = (ins "size_t":$n, "::mlir::Type":$t);
  let genVerifyInvariantsDecl = 1;
}

def SeqTypeConstraint : Type<CPred<"$_self.isa<SeqType>()">, "Aetherling Sequence type">;