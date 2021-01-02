//===- Types.td - Sequence Language data type definitions --*- tablegen -*-===//
//
// Basic data types for the RTL dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class SeqLangType<string name> : TypeDef<SequenceLanguage_Dialect, name> { }

// A simple fixed size array.
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