#include "casm/crystallography/UnitCellCoordRep.hh"

#include "casm/crystallography/Lattice.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/misc/CASM_Eigen_math.hh"

namespace CASM {
namespace xtal {

/// \brief Make UnitCellCoordRep
///
/// \param op The SymOp to make a UnitCellCoordRep for
/// \param lattice The lattice
/// \param symop_site_map The sites that op transforms each
///     basis site to, as constructed by `xtal::symop_site_map`.
UnitCellCoordRep make_unitcellcoord_rep(
    SymOp const &op, Lattice const &lattice,
    std::vector<UnitCellCoord> const &symop_site_map) {
  UnitCellCoordRep rep;
  rep.point_matrix = lround(cart2frac(op.matrix, lattice));
  for (UnitCellCoord const &site : symop_site_map) {
    rep.sublattice_index.push_back(site.sublattice());
    rep.unitcell_indices.push_back(site.unitcell());
  }
  return rep;
}

/// \brief Apply symmetry to UnitCellCoord
///
/// \param rep UnitCellCoordRep representation of the symmetry operation
/// \param integral_site_coordinate Coordinate being transformed
///
UnitCellCoord &apply(UnitCellCoordRep const &rep,
                     UnitCellCoord &integral_site_coordinate) {
  integral_site_coordinate = copy_apply(rep, integral_site_coordinate);
  return integral_site_coordinate;
}

/// \brief Apply symmetry to UnitCellCoord
///
/// \param rep UnitCellCoordRep representation of the symmetry operation
/// \param integral_site_coordinate Coordinate being transformed
///
UnitCellCoord copy_apply(UnitCellCoordRep const &rep,
                         UnitCellCoord integral_site_coordinate) {
  UnitCell unitcell_indices =
      rep.point_matrix * integral_site_coordinate.unitcell() +
      rep.unitcell_indices[integral_site_coordinate.sublattice()];
  Index sublattice_index =
      rep.sublattice_index[integral_site_coordinate.sublattice()];
  return UnitCellCoord(sublattice_index, unitcell_indices);
}

}  // namespace xtal
}  // namespace CASM
