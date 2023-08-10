#include "casm/crystallography/version.hh"

using namespace CASM;
using namespace CASM::xtal;

#ifndef CASM_XTAL_TXT_VERSION
#define CASM_XTAL_TXT_VERSION "unknown"
#endif

const std::string &CASM::xtal::version() {
  static const std::string &ver = CASM_XTAL_TXT_VERSION;
  return ver;
};
