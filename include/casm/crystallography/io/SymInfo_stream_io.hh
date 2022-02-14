#ifndef CASM_xtal_SymInfo_stream_io
#define CASM_xtal_SymInfo_stream_io

#include <iomanip>
#include <iostream>
#include <string>

#include "casm/casm_io/enum/stream_io.hh"
#include "casm/crystallography/SymInfo.hh"
#include "casm/global/definitions.hh"

namespace CASM {

class Log;

namespace xtal {

struct SymInfo;

}

ENUM_TRAITS(xtal::symmetry_type)
ENUM_IO_DECL(xtal::symmetry_type)

namespace xtal {

/// Options for printing SymInfo
struct SymInfoOptions {
  SymInfoOptions(COORD_TYPE _coord_type = FRAC, double _tol = TOL,
                 Index _prec = 7, bool _print_matrix_tau = false)
      : coord_type(_coord_type),
        tol(_tol),
        prec(_prec),
        print_matrix_tau(_print_matrix_tau) {}
  COORD_TYPE coord_type;
  double tol;
  Index prec;
  bool print_matrix_tau;
};

/// \brief Print SymInfo
void print_sym_info(Log &log, const SymInfo &info,
                    SymInfoOptions opt = SymInfoOptions());

/// \brief Print SymInfo to string
std::string to_string(const SymInfo &info,
                      SymInfoOptions opt = SymInfoOptions());

/// \brief Print symmetry symbol to string
std::string to_brief_unicode(const SymInfo &info,
                             SymInfoOptions opt = SymInfoOptions());

/// \brief Print SymInfo to string
std::string description(const SymOp &op, const xtal::Lattice &lat,
                        SymInfoOptions opt = SymInfoOptions());

/// \brief Print SymInfo to brief string
std::string brief_description(const SymOp &op, const xtal::Lattice &lat,
                              SymInfoOptions opt = SymInfoOptions());

}  // namespace xtal
}  // namespace CASM

#endif
