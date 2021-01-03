set(LLVM_OPTIONAL_SOURCES
  DialectRegistration.cpp
  mlir_std_runner.cpp
  Ops.cpp
  )

file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTSequenceLanguage
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SequenceLanguage

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR

  LINK_COMPONENTS
  Support

  DEPENDS
  MLIRSequenceLanguageIncGen
)

add_dependencies(circt-headers MLIRSequenceLanguageIncGen)
