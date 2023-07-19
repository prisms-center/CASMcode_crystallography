# see /opt/local/share/cmake-3.16/Modules/FindGTest.cmake
find_package(GTest QUIET)

# NOTE: the upper case GTEST! CK
if(NOT GTEST_FOUND)
  # Download and unpack googletest at configure time
  # but only if needed! CK
  configure_file(${CMAKE_SOURCE_DIR}/cmake/GoogleTest-CMakeLists.txt.in
    ${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()

  # Add googletest directly to our build. This defines
  # the gtest and gtest_main targets.
  add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
    ${CMAKE_BINARY_DIR}/googletest-build
    EXCLUDE_FROM_ALL
  )
endif()

# create libcasm_testing
set(
  libcasm_testing_SOURCES
  ${PROJECT_SOURCE_DIR}/tests/unit/Common.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/autotools.cc
  ${PROJECT_SOURCE_DIR}/tests/unit/testdir.cc
)
add_library(casm_testing SHARED ${libcasm_testing_SOURCES})
target_include_directories(casm_testing
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/unit>
)
target_link_libraries(casm_testing
  gtest
  CASM::casm_global
  casm_crystallography
)
target_compile_options(casm_testing
  PUBLIC
    "-DABS_SRCDIR=\"${CMAKE_SOURCE_DIR}\""
    "-DABS_TOP_BUILDDIR=\"${CMAKE_BINARY_DIR}\""
)

################################################################
# casm_unit_crystallography
add_executable(casm_unit_crystallography
  ${PROJECT_SOURCE_DIR}/tests/unit/gtest_main_run_all.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/Site_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/BasicStructureTools_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/SymInfo_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/SymRepBuilder_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/SimpleStructureIO_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/LinearIndexConverter_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/SuperlatticeEnumerator_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/Lattice_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/SymType_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/BasicStructureIO_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/Coordinate_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/AnisoValTraits_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/HermiteCounter_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/BasicStructure_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/version_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/IntegralCoordinateWithin_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/Niggli_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/SimpleStructure_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/Molecule_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/UnitCellCoord_test.cpp
  ${PROJECT_SOURCE_DIR}/tests/unit/crystallography/FactorGroupConsistency_test.cpp
)
target_compile_features(casm_unit_crystallography PRIVATE cxx_auto_type)
target_link_libraries(casm_unit_crystallography
  gtest
  CASM::casm_global
  casm_crystallography
  casm_testing
)
target_include_directories(casm_unit_crystallography
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/unit>
)

add_test(NAME casm_unit_crystallography COMMAND casm_unit_crystallography)
