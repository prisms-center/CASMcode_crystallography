#ifndef CASM_crystallography_version
#define CASM_crystallography_version

#include <string>

namespace CASM {
namespace xtal {

const std::string &version();  // Returns the version defined by the TXT_VERSION
                               // macro at compile time

}
}  // namespace CASM

extern "C" {

/// \brief Return the libcasm_crystallography version number
inline const char *casm_crystallography_version() {
  return CASM::xtal::version().c_str();
}
}

#endif
