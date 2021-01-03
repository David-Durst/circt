// RUN: circt-opt %s | FileCheck %s

// CHECK-LABEL: func @test1(%arg0: !sequencelanguage.seq<4xi4>, %arg1: !sequencelanguage.seq<4xi4>) -> !sequencelanguage.seq<4xi4> {
func @test1(%arg0: !sequencelanguage.seq<4xi4>, %arg1: !sequencelanguage.seq<4xi4>) -> !sequencelanguage.seq<4xi4> {
  %a = sequencelanguage.map2(%arg0: !sequencelanguage.seq<4xi4>, %arg1: !sequencelanguage.seq<4xi4>, !sequencelanguage.add : i4 -> i4) : !sequencelanguage.seq<4xi4>
  return %a : !sequencelanguage.seq<4xi4>
}
// CHECK-NEXT:  }
