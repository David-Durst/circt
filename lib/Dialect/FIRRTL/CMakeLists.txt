file(GLOB globbed *.cpp)
add_circt_dialect_library(MLIRFIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/FIRRTL

  DEPENDS
  MLIRFIRRTLIncGen
  MLIRFIRRTLEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  )

add_dependencies(circt-headers MLIRFIRRTLIncGen MLIRFIRRTLEnumsIncGen)
 