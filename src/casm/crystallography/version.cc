#include "casm/crystallography/version.hh"

using namespace CASM;
using namespace CASM::xtal;

#ifndef TXT_VERSION
#define TXT_VERSION "unknown"
#endif

const std::string &CASM::xtal::version() {
  static const std::string &ver = TXT_VERSION;
  return ver;
};
