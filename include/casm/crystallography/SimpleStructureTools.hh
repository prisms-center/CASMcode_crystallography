#ifndef SIMPLESTRUCTURETOOLS_HH
#define SIMPLESTRUCTURETOOLS_HH

#include <map>
#include <set>
#include <string>
#include <vector>

#include "casm/crystallography/DoFDecl.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/external/Eigen/Dense"
#include "casm/global/definitions.hh"

namespace CASM {

namespace xtal {

/** \ingroup Structure
 *  @{
 */

// TODO: Move this into crystallography declarations or something of the sort
namespace SimpleStructureTools {
// Defined in SimpleStructure.hh
enum class SpeciesMode;
}  // namespace SimpleStructureTools

class SimpleStructure;
class Site;
class Molecule;
class BasicStructure;

SimpleStructure make_superstructure(Eigen::Ref<const Eigen::Matrix3i> const &_T,
                                    SimpleStructure const &_sstruc);

/// \brief Constructs a vector containing the basis index of the ith site in the
/// supercell
std::vector<Index> superstructure_basis_idx(
    Eigen::Ref<const Eigen::Matrix3i> const &_T,
    SimpleStructure const &_sstruc);

/// \brief Construct from decorated structure
SimpleStructure make_simple_structure(BasicStructure const &_struc);

/// \brief Construct SimpleStructure from poscar stream
SimpleStructure make_simple_structure(std::istream &poscar_stream,
                                      double tol = TOL);

/// \brief Determine which sites of a BasicStructure can host each atom of a
/// SimpleStructure result[i] is set of site indices in @param _prim that can
/// host atom 'i' of @param sstruc
std::vector<std::set<Index>> atom_site_compatibility(
    SimpleStructure const &sstruc, BasicStructure const &_prim);

/// \brief Determine which sites of a BasicStructure can host each molecule of a
/// SimpleStructure result[i] is set of site indices in @param _prim that can
/// host molecule 'i' of @param sstruc
std::vector<std::set<Index>> mol_site_compatibility(
    SimpleStructure const &sstruc, BasicStructure const &_prim);

/// \brief use information in _reference to initialize atom_info from mol_info
void _atomize(SimpleStructure &_sstruc,
              Eigen::Ref<const Eigen::VectorXi> const &_mol_occ,
              BasicStructure const &_reference);

/// \brief Construct BasicStructure from SimpleStructure.
/// @param _sstruc SimpleStructure used as source data for conversion
/// @param _all_dofs holds names of additional DOFs to initialize in structure
/// @param mode specifies whether ATOM or MOL info of _sstruc should be used to
/// build sites of structure
/// @param _allowed_occupants List of allowed molecules at each site; if empty,
/// occupants are assumed to be atoms
///        having the species names and attributes indicated by _sstruc
BasicStructure make_basic_structure(
    SimpleStructure const &_sstruc, std::vector<DoFKey> const &_all_dofs,
    SimpleStructureTools::SpeciesMode mode,
    std::vector<std::vector<Molecule>> _allowed_occupants = {});

std::vector<Eigen::MatrixXd> generate_invariant_shuffle_modes(
    const std::vector<xtal::SymOp> &factor_group,
    const std::vector<Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic,
                                               Index>> &permute_group);

/// \brief Transform local or global properties
std::map<std::string, Eigen::MatrixXd> &apply(
    xtal::SymOp const &op, std::map<std::string, Eigen::MatrixXd> &properties);

/// \brief Copy and transform local or global properties
std::map<std::string, Eigen::MatrixXd> copy_apply(
    xtal::SymOp const &op,
    std::map<std::string, Eigen::MatrixXd> const &properties);

/// \brief Transform global properties
std::map<std::string, Eigen::VectorXd> &apply(
    xtal::SymOp const &op, std::map<std::string, Eigen::VectorXd> &properties);

/// \brief Copy and transform global properties
std::map<std::string, Eigen::VectorXd> copy_apply(
    xtal::SymOp const &op,
    std::map<std::string, Eigen::VectorXd> const &properties);

/// \brief Transform a SimpleStructure
SimpleStructure apply(xtal::SymOp const &op, SimpleStructure &sstruc);

/// \brief Copy and transform a SimpleStructure
SimpleStructure copy_apply(xtal::SymOp const &op, SimpleStructure sstruc);

/// \brief Check if two structures are equivalent
bool is_equivalent(SimpleStructure const &first, SimpleStructure const &second,
                   double xtal_tol = TOL,
                   std::map<std::string, double> properties_tol =
                       std::map<std::string, double>());

/** @} */
}  // namespace xtal

}  // namespace CASM
#endif
