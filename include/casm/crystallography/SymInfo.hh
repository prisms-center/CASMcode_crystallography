#ifndef CASM_xtal_SymInfo
#define CASM_xtal_SymInfo

#include "casm/crystallography/Coordinate.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/external/Eigen/Dense"

namespace CASM {
namespace xtal {

enum class symmetry_type {
  identity_op,
  mirror_op,
  glide_op,
  rotation_op,
  screw_op,
  inversion_op,
  rotoinversion_op,
  invalid_op
};

/// \brief Symmetry operation information
struct SymInfo {
  SymInfo(SymOp const &op, xtal::Lattice const &lat);

  SymInfo(SymInfo const &sym_info);

  /// The lattice used for coordinates
  xtal::Lattice lattice;

  /// One of: identity_op, mirror_op, glide_op, rotation_op, screw_op,
  ///         inversion_op, rotoinversion_op, or invalid_op
  symmetry_type op_type;

  /// Rotation axis if operation S is rotation/screw operation
  /// If improper operation, rotation axis of inversion*S
  /// (implying that axis is normal vector for a mirror plane)
  /// normalized to length 1
  /// axis is zero if operation is identity or inversion
  xtal::Coordinate axis;

  /// Rotation angle, if operation S is rotation/screw operation
  /// If improper operation, rotation angle of inversion*S
  double angle;

  /// Component of tau parallel to 'axis' (for rotation)
  /// or perpendicular to 'axis', for mirror operation
  xtal::Coordinate screw_glide_shift;

  /// A Cartesian coordinate that is invariant to the operation (if one exists)
  xtal::Coordinate location;

  /// If time reversal symmetry
  bool time_reversal;

 private:
  void _set(Eigen::Vector3d const &_axis,
            Eigen::Vector3d const &_screw_glide_shift,
            Eigen::Vector3d const &_location, xtal::Lattice const &lat);
};

}  // namespace xtal
}  // namespace CASM

#endif
