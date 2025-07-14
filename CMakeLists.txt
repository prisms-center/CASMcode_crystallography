cmake_minimum_required(VERSION 3.20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15" CACHE STRING "Minimum OS X deployment version")

project(CASMcode_crystallography VERSION 2.2.0 LANGUAGES CXX)

# set CMAKE_INSTALL_X variables
include(GNUInstallDirs)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# try to use ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

##############################################
## Find dependencies

# Should find ZLIB::ZLIB
find_package(ZLIB)

# Find CASM
if(NOT DEFINED CASM_PREFIX)
  message(STATUS "CASM_PREFIX not defined")
  # try to find Python
  find_package (Python COMPONENTS Interpreter Development)
  if(DEFINED Python_EXECUTABLE)
    # if Python found, obtain CASM_PREFIX from the libcasm.casmglobal
    message(STATUS "found Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "checking for libcasm-global")
    execute_process(
      COMMAND pip show libcasm-global
      RESULT_VARIABLE EXIT_CODE
      OUTPUT_QUIET
    )
    if (${EXIT_CODE} EQUAL 0)
      message(STATUS "found libcasm-global")
      execute_process(COMMAND ${Python_EXECUTABLE} -m libcasm.casmglobal --prefix
                      OUTPUT_VARIABLE CASM_PREFIX_RAW)
      string(STRIP ${CASM_PREFIX_RAW} CASM_PREFIX)
      message(STATUS "CASM_PREFIX: ${CASM_PREFIX}")
    else()
      message(STATUS "did not find libcasm-global")
    endif()
  endif()
endif()
if(DEFINED CASM_PREFIX)
  set(CASMcode_global_ROOT ${CASM_PREFIX}/share/CASMcode_global/cmake)
endif()
find_package(CASMcode_global)
if(NOT CASMcode_global_FOUND)
  message(FATAL_ERROR "CMake failed to find CASMcode_global")
endif()
# if successful, we have CASM::casm_global

# if no user CMAKE_INSTALL_PREFIX, use CASM_PREFIX if it exists
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(DEFINED CASM_PREFIX)
    message(STATUS "CMAKE_INSTALL_PREFIX initialized to default, so updating CMAKE_INSTALL_PREFIX to CASM_PREFIX")
    set(CMAKE_INSTALL_PREFIX ${CASM_PREFIX} CACHE PATH "set CMAKE_INSTALL_PREFIX to CASM_PREFIX" FORCE)
    message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
  endif()
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

##############################################
## Build libcasm_crystallography

# create libcasm_crystallography
set(
  libcasm_crystallography_HEADERS
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/OccupantDoFIsEquivalent.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/BasicStructureTools.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Lattice.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/StrucMapCalculatorInterface.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Site.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/CanonicalForm.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SimpleStructure.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/HermiteCounter.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SimpleStrucMapCalculator.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/StrainConverter.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/DoFSet.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/DoFDecl.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SymInfo.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/LinearIndexConverter.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/AnisoValTraits.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Superlattice.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/LatticeMap.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/BasicStructure.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/IntegralCoordinateWithin.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Strain.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/UnitCellCoord.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/LatticeIsEquivalent.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Molecule.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/version.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/UnitCellCoordRep.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SymTools.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SuperlatticeEnumerator.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/StrucMapping.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Coordinate.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SimpleStructureTools.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/UnitCellCoordTraits.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Niggli.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SymRepBuilder.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/CoordinateSystems.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SymTypeComparator.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SpeciesProperty.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/SymType.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/Adapter.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/SimpleStructureIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/DoFSetIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/SymInfo_stream_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/LatticeIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/SuperlatticeEnumeratorIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/BasicStructureIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/UnitCellCoordIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/VaspIO.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/SymInfo_json_io.hh
  ${PROJECT_SOURCE_DIR}/include/casm/crystallography/io/jsonStruc.hh
)
set(
  libcasm_crystallography_SOURCES
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Superlattice.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/AnisoValTraits.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/LatticeMap.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/BasicStructure.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/IntegralCoordinateWithin.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Strain.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/UnitCellCoord.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/DoFSet.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/StrainConverter.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SymInfo.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/LinearIndexConverter.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/CanonicalForm.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Site.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SimpleStructure.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SimpleStrucMapCalculator.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/HermiteCounter.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/OccupantDoFIsEquivalent.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/BasicStructureTools.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Lattice.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/CoordinateSystems.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SymTypeComparator.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SpeciesProperty.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SymType.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SimpleStructureTools.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Coordinate.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/UnitCellCoordTraits.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Niggli.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SuperlatticeEnumerator.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/SymTools.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/StrucMapping.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/LatticeIsEquivalent.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/Molecule.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/version.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/UnitCellCoordRep.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/SuperlatticeEnumeratorIO.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/LatticeIO.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/BasicStructureIO.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/SimpleStructureIO.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/DoFSetIO.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/SymInfo_stream_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/SymInfo_json_io.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/VaspIO.cc
  ${PROJECT_SOURCE_DIR}/src/casm/crystallography/io/UnitCellCoordIO.cc
)
add_library(casm_crystallography SHARED ${libcasm_crystallography_SOURCES})
target_include_directories(casm_crystallography
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/casm/external>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/casm/external/gzstream>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/casm/external>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/casm/external/gzstream>
)
target_compile_options(casm_crystallography
  PUBLIC
    "-DCASM_XTAL_TXT_VERSION=\"${CMAKE_PROJECT_VERSION}\""
    -DEIGEN_DEFAULT_DENSE_INDEX_TYPE=long
    -DGZSTREAM_NAMESPACE=gz
)
target_link_libraries(casm_crystallography
  ZLIB::ZLIB
  ${CMAKE_DL_LIBS}
  CASM::casm_global
)
if(APPLE)
  set_target_properties(
    casm_crystallography PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(
    casm_crystallography PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

##############################################
## Install libcasm_crystallography

# install header files in <prefix>/libcasm/include/,
# while preserving directory structure
foreach ( filevar ${libcasm_crystallography_HEADERS} )
  file(RELATIVE_PATH relfile ${PROJECT_SOURCE_DIR}/include/ ${filevar})
  get_filename_component( reldir ${relfile} DIRECTORY )
  install( FILES ${filevar} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${reldir} )
endforeach()

# install libcasm_crystallography in <prefix>/libcasm/lib/
install(
  TARGETS casm_crystallography
  EXPORT CASMcode_crystallographyTargets
  DESTINATION lib)

##############################################
## Python extensions

# The CMake package config and target files are installed under the Python
# package root. This is necessary to ensure that all the relative paths in the
# helloTargets.cmake resolve correctly. It also provides encapsulation.
#
# The actual path used must be selected so that consuming projects can locate it
# via `find_package`. To support finding CMake packages in the Python package
# prefix, using `find_package`s default search path of
# `<prefix>/<name>/share/<name>*/cmake/` is reasonable. Adding the Python
# package installation prefix to CMAKE_PREFIX_PATH in combination with this path
# will allow `find_package` to find this package and any other package installed
# via a Python package if the CMake and Python packages are named the same.
set(CASM_CRYSTALLOGRAPHY_CMAKE_PACKAGE_INSTALL_SUBDIR "share/CASMcode_crystallography/cmake")

install(
  EXPORT CASMcode_crystallographyTargets
  NAMESPACE CASM::
  DESTINATION ${CASM_CRYSTALLOGRAPHY_CMAKE_PACKAGE_INSTALL_SUBDIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  CASMcode_crystallographyConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/CASMcode_crystallographyConfig.cmake.in" CASMcode_crystallographyConfig.cmake
  INSTALL_DESTINATION ${CASM_CRYSTALLOGRAPHY_CMAKE_PACKAGE_INSTALL_SUBDIR})

install(FILES "${PROJECT_BINARY_DIR}/CASMcode_crystallographyConfig.cmake"
              "${PROJECT_BINARY_DIR}/CASMcode_crystallographyConfigVersion.cmake"
        DESTINATION ${CASM_CRYSTALLOGRAPHY_CMAKE_PACKAGE_INSTALL_SUBDIR})

# We are using the SKBUILD variable, which is defined when scikit-build is
# running the CMake build, to control building the Python wrapper. This allows
# the C++ project to be installed, standalone, when using the standard CMake
# build flow.
if(DEFINED SKBUILD)

  # call pybind11-config to obtain the root of the cmake package
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
                  OUTPUT_VARIABLE pybind11_ROOT_RAW)
  string(STRIP ${pybind11_ROOT_RAW} pybind11_ROOT)
  find_package(pybind11)

  # The extension modules must load:
  # - the casm_global library
  # - the casm_crystallography library
  # They can be found by setting a relative rpath

  ### libcasm.xtal._xtal ###
  pybind11_add_module(_xtal MODULE
                      "${PROJECT_SOURCE_DIR}/python/src/xtal.cpp")
  target_link_libraries(_xtal PRIVATE CASM::casm_global)
  target_link_libraries(_xtal PRIVATE casm_crystallography)
  install(TARGETS _xtal DESTINATION xtal)
  if(APPLE)
    set_target_properties(
      _xtal PROPERTIES INSTALL_RPATH "@loader_path/../lib")
  else()
    set_target_properties(
      _xtal PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
  endif()

endif()
