#ifndef COORDINATESYSTEMS_HH
#define COORDINATESYSTEMS_HH

#include <iostream>
#include <string>

#include "casm/global/enum.hh"

namespace CASM {
namespace xtal {

/** \ingroup Coordinate
 *  @{
 */

/// Return the name of a coordinate mode
inline std::string coord_mode_name(COORD_TYPE mode) {
  if (mode == FRAC) return "Direct";
  if (mode == CART) return "Cartesian";
  return "Unknown";
}

/** @} */

}  // namespace xtal
};  // namespace CASM
#endif
