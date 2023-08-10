#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "pybind11_json/pybind11_json.hpp"

// CASM
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/CanonicalForm.hh"
#include "casm/crystallography/Lattice.hh"
#include "casm/crystallography/LatticeIsEquivalent.hh"
#include "casm/crystallography/LinearIndexConverter.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/SimpleStructureTools.hh"
#include "casm/crystallography/StrainConverter.hh"
#include "casm/crystallography/SuperlatticeEnumerator.hh"
#include "casm/crystallography/SymInfo.hh"
#include "casm/crystallography/SymTools.hh"
#include "casm/crystallography/UnitCellCoord.hh"
#include "casm/crystallography/UnitCellCoordRep.hh"
#include "casm/crystallography/io/BasicStructureIO.hh"
#include "casm/crystallography/io/SimpleStructureIO.hh"
#include "casm/crystallography/io/SymInfo_json_io.hh"
#include "casm/crystallography/io/SymInfo_stream_io.hh"
#include "casm/crystallography/io/VaspIO.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

namespace _xtal_impl {

Eigen::MatrixXd pseudoinverse(Eigen::MatrixXd const &M) {
  Index dim = M.rows();
  return M.transpose()
      .colPivHouseholderQr()
      .solve(Eigen::MatrixXd::Identity(dim, dim))
      .transpose();
}
}  // namespace _xtal_impl

// xtal

double default_tol() { return TOL; }

// Lattice

xtal::Lattice make_canonical_lattice(xtal::Lattice lattice) {
  lattice.make_right_handed();
  return xtal::canonical::equivalent(lattice);
}

/// \brief Convert fractional coordinates to Cartesian coordinates
///
/// \param lattice Lattice
/// \param coordinate_frac Fractional coordinates, as columns of a matrix
Eigen::MatrixXd fractional_to_cartesian(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &coordinate_frac) {
  return lattice.lat_column_mat() * coordinate_frac;
}

/// \brief Convert Cartesian coordinates to fractional coordinates
///
/// \param lattice Lattice
/// \param coordinate_cart Cartesian coordinates, as columns of a matrix
Eigen::MatrixXd cartesian_to_fractional(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &coordinate_cart) {
  return lattice.inv_lat_column_mat() * coordinate_cart;
}

/// \brief Translate fractional coordinates within the lattice unit cell
///
/// \param lattice Lattice
/// \param coordinate_frac Fractional coordinates, as columns of a matrix
Eigen::MatrixXd fractional_within(xtal::Lattice const &lattice,
                                  Eigen::MatrixXd coordinate_frac) {
  double tshift;
  for (Index col = 0; col < coordinate_frac.cols(); ++col) {
    for (Index i = 0; i < 3; i++) {
      tshift = floor(coordinate_frac(i, col) + 1E-6);
      if (!almost_zero(tshift, TOL)) {
        coordinate_frac(i, col) -= tshift;
      }
    }
  }
  return coordinate_frac;
}

std::vector<xtal::SymOp> make_lattice_point_group(
    xtal::Lattice const &lattice) {
  return xtal::make_point_group(lattice);
}

std::vector<xtal::Lattice> enumerate_superlattices(
    xtal::Lattice const &unit_lattice,
    std::vector<xtal::SymOp> const &point_group, Index max_volume,
    Index min_volume = 1, std::string dirs = std::string("abc")) {
  xtal::ScelEnumProps enum_props{min_volume, max_volume + 1, dirs};
  xtal::SuperlatticeEnumerator enumerator{unit_lattice, point_group,
                                          enum_props};
  std::vector<xtal::Lattice> superlattices;
  for (auto const &superlat : enumerator) {
    superlattices.push_back(
        xtal::canonical::equivalent(superlat, point_group, unit_lattice.tol()));
  }
  return superlattices;
}

bool lattice_is_equivalent_to(xtal::Lattice const &lattice1,
                              xtal::Lattice const &lattice2) {
  return xtal::is_equivalent(lattice1, lattice2);
}

std::pair<bool, Eigen::Matrix3d> is_superlattice_of(
    xtal::Lattice const &superlattice, xtal::Lattice const &unit_lattice) {
  double tol = std::max(superlattice.tol(), unit_lattice.tol());
  return xtal::is_superlattice(superlattice, unit_lattice, tol);
}

Eigen::Matrix3l make_transformation_matrix_to_super(
    xtal::Lattice const &superlattice, xtal::Lattice const &unit_lattice) {
  double tol = std::max(superlattice.tol(), unit_lattice.tol());
  return xtal::make_transformation_matrix_to_super(unit_lattice, superlattice,
                                                   tol);
}

/// \brief Check if S = point_group[point_group_index] * L * T, with integer T
///
/// \returns (is_equivalent, T, point_group_index)
std::tuple<bool, Eigen::MatrixXd, Index> is_equivalent_superlattice_of(
    xtal::Lattice const &superlattice, xtal::Lattice const &unit_lattice,
    std::vector<xtal::SymOp> const &point_group = std::vector<xtal::SymOp>{}) {
  double tol = std::max(superlattice.tol(), unit_lattice.tol());
  auto result = is_equivalent_superlattice(
      superlattice, unit_lattice, point_group.begin(), point_group.end(), tol);
  bool is_equivalent = (result.first != point_group.end());
  Index point_group_index = -1;
  if (is_equivalent) {
    point_group_index = std::distance(point_group.begin(), result.first);
  }
  return std::tuple<bool, Eigen::MatrixXd, Index>(is_equivalent, result.second,
                                                  point_group_index);
}

xtal::Lattice make_superduperlattice(
    std::vector<xtal::Lattice> const &lattices,
    std::string mode = std::string("commensurate"),
    std::vector<xtal::SymOp> const &point_group = std::vector<xtal::SymOp>{}) {
  if (mode == "commensurate") {
    return xtal::make_commensurate_superduperlattice(lattices.begin(),
                                                     lattices.end());
  } else if (mode == "minimal_commensurate") {
    return xtal::make_minimal_commensurate_superduperlattice(
        lattices.begin(), lattices.end(), point_group.begin(),
        point_group.end());
  } else if (mode == "fully_commensurate") {
    return xtal::make_fully_commensurate_superduperlattice(
        lattices.begin(), lattices.end(), point_group.begin(),
        point_group.end());
  } else {
    std::stringstream msg;
    msg << "Error in make_superduperlattice: Unrecognized mode=" << mode;
    throw std::runtime_error(msg.str());
  }
}

// DoFSetBasis

struct DoFSetBasis {
  DoFSetBasis(
      std::string const &_dofname,
      std::vector<std::string> const &_axis_names = std::vector<std::string>{},
      Eigen::MatrixXd const &_basis = Eigen::MatrixXd(0, 0))
      : dofname(_dofname), axis_names(_axis_names), basis(_basis) {
    if (Index(axis_names.size()) != basis.cols()) {
      throw std::runtime_error(
          "Error in DoFSetBasis::DoFSetBasis(): axis_names.size() != "
          "basis.cols()");
    }
    if (axis_names.size() == 0) {
      axis_names = CASM::AnisoValTraits(dofname).standard_var_names();
      Index dim = axis_names.size();
      basis = Eigen::MatrixXd::Identity(dim, dim);
    }
  }

  /// The type of DoF
  std::string dofname;

  /// A name for each basis vector (i.e. for each column of basis).
  std::vector<std::string> axis_names;

  /// Basis vectors, as columns of a matrix, such that `x_standard = basis *
  /// x_prim`. If `basis.cols() == 0`, the standard basis will be used when
  /// constructing a prim.
  Eigen::MatrixXd basis;
};

std::string get_dofsetbasis_dofname(DoFSetBasis const &dofsetbasis) {
  return dofsetbasis.dofname;
}
std::vector<std::string> get_dofsetbasis_axis_names(
    DoFSetBasis const &dofsetbasis) {
  return dofsetbasis.axis_names;
}

Eigen::MatrixXd get_dofsetbasis_basis(DoFSetBasis const &dofsetbasis) {
  return dofsetbasis.basis;
}

/// \brief Construct DoFSetBasis
///
/// \param dofname DoF name. Must be a CASM-supported DoF type.
/// \param axis_names DoFSet axis names. Size equals number of columns in basis.
/// \param basis Basis vectors, as columns of a matrix, such that `x_standard =
/// basis * x_prim`. If `basis.cols() == 0`, the standard basis will be used.
///
DoFSetBasis make_dofsetbasis(
    std::string dofname,
    std::vector<std::string> const &axis_names = std::vector<std::string>{},
    Eigen::MatrixXd const &basis = Eigen::MatrixXd(0, 0)) {
  return DoFSetBasis(dofname, axis_names, basis);
}

// SpeciesProperty -> properties

std::map<std::string, xtal::SpeciesProperty> make_species_properties(
    std::map<std::string, Eigen::MatrixXd> species_properties) {
  std::map<std::string, xtal::SpeciesProperty> result;
  for (auto const &pair : species_properties) {
    result.emplace(pair.first, xtal::SpeciesProperty{AnisoValTraits(pair.first),
                                                     pair.second});
  }
  return result;
}

// AtomComponent

xtal::AtomPosition make_atom_position(
    std::string name, Eigen::Vector3d pos,
    std::map<std::string, Eigen::MatrixXd> properties = {}) {
  xtal::AtomPosition atom(pos, name);
  atom.set_properties(make_species_properties(properties));
  return atom;
}

std::map<std::string, Eigen::MatrixXd> get_atom_position_properties(
    xtal::AtomPosition const &atom) {
  std::map<std::string, Eigen::MatrixXd> result;
  for (auto const &pair : atom.properties()) {
    result.emplace(pair.first, pair.second.value());
  }
  return result;
}

// Occupant

xtal::Molecule make_molecule(
    std::string name, std::vector<xtal::AtomPosition> atoms = {},
    bool divisible = false,
    std::map<std::string, Eigen::MatrixXd> properties = {}) {
  xtal::Molecule mol(name, atoms, divisible);
  mol.set_properties(make_species_properties(properties));
  return mol;
}

std::map<std::string, Eigen::MatrixXd> get_molecule_properties(
    xtal::Molecule const &mol) {
  std::map<std::string, Eigen::MatrixXd> result;
  for (auto const &pair : mol.properties()) {
    result.emplace(pair.first, pair.second.value());
  }
  return result;
}

// Prim

std::shared_ptr<xtal::BasicStructure> make_prim(
    xtal::Lattice const &lattice, Eigen::MatrixXd const &coordinate_frac,
    std::vector<std::vector<std::string>> const &occ_dof,
    std::vector<std::vector<DoFSetBasis>> const &local_dof =
        std::vector<std::vector<DoFSetBasis>>{},
    std::vector<DoFSetBasis> const &global_dof = std::vector<DoFSetBasis>{},
    std::map<std::string, xtal::Molecule> const &molecules =
        std::map<std::string, xtal::Molecule>{},
    std::string title = std::string("prim")) {
  // validation
  if (coordinate_frac.rows() != 3) {
    throw std::runtime_error("Error in make_prim: coordinate_frac.rows() != 3");
  }
  if (coordinate_frac.cols() != Index(occ_dof.size())) {
    throw std::runtime_error(
        "Error in make_prim: coordinate_frac.cols() != "
        "occ_dof.size()");
  }
  if (local_dof.size() && coordinate_frac.cols() != Index(local_dof.size())) {
    throw std::runtime_error(
        "Error in make_prim: local_dof.size() && "
        "coordinate_frac.cols() != occ_dof.size()");
  }

  // construct prim
  auto shared_prim = std::make_shared<xtal::BasicStructure>(lattice);
  xtal::BasicStructure &prim = *shared_prim;
  prim.set_title(title);

  // set basis sites
  for (Index b = 0; b < coordinate_frac.cols(); ++b) {
    xtal::Coordinate coord{coordinate_frac.col(b), prim.lattice(), FRAC};
    std::vector<xtal::Molecule> site_occ;
    for (std::string label : occ_dof[b]) {
      if (molecules.count(label)) {
        site_occ.push_back(molecules.at(label));
      } else {
        site_occ.push_back(xtal::Molecule{label});
      }
    }

    std::vector<xtal::SiteDoFSet> site_dofsets;
    if (local_dof.size()) {
      for (auto const &dofsetbasis : local_dof[b]) {
        if (dofsetbasis.basis.cols()) {
          site_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname),
                                    dofsetbasis.axis_names, dofsetbasis.basis,
                                    std::unordered_set<std::string>{});
        } else {
          site_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname));
        }
      }
    }

    xtal::Site site{coord, site_occ, site_dofsets};
    prim.push_back(site, FRAC);
  }
  prim.set_unique_names(occ_dof);

  // set global dof
  std::vector<xtal::DoFSet> global_dofsets;
  for (auto const &dofsetbasis : global_dof) {
    if (dofsetbasis.basis.cols()) {
      global_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname),
                                  dofsetbasis.axis_names, dofsetbasis.basis);
    } else {
      global_dofsets.emplace_back(AnisoValTraits(dofsetbasis.dofname));
    }
  }

  prim.set_global_dofs(global_dofsets);

  return shared_prim;
}

void init_prim(
    xtal::BasicStructure &obj, xtal::Lattice const &lattice,
    Eigen::MatrixXd const &coordinate_frac,
    std::vector<std::vector<std::string>> const &occ_dof,
    std::vector<std::vector<DoFSetBasis>> const &local_dof =
        std::vector<std::vector<DoFSetBasis>>{},
    std::vector<DoFSetBasis> const &global_dof = std::vector<DoFSetBasis>{},
    std::map<std::string, xtal::Molecule> const &molecules =
        std::map<std::string, xtal::Molecule>{},
    std::string title = std::string("prim")) {
  auto prim = make_prim(lattice, coordinate_frac, occ_dof, local_dof,
                        global_dof, molecules, title);
  new (&obj) xtal::BasicStructure(*prim);
}

/// \brief Construct xtal::BasicStructure from JSON string
std::shared_ptr<xtal::BasicStructure const> prim_from_json(
    std::string const &prim_json_str, double xtal_tol) {
  jsonParser json = jsonParser::parse(prim_json_str);
  ParsingDictionary<AnisoValTraits> const *aniso_val_dict = nullptr;
  return std::make_shared<xtal::BasicStructure>(
      read_prim(json, xtal_tol, aniso_val_dict));
}

/// \brief Construct xtal::BasicStructure from poscar stream
std::shared_ptr<xtal::BasicStructure const> prim_from_poscar_stream(
    std::istream &poscar_stream,
    std::vector<std::vector<std::string>> const &occ_dof = {},
    double xtal_tol = TOL) {
  auto prim = std::make_shared<xtal::BasicStructure>(
      xtal::BasicStructure::from_poscar_stream(poscar_stream, xtal_tol));
  if (occ_dof.size() == 0) {
    return prim;
  }

  Eigen::MatrixXd frac_coords(3, prim->basis().size());
  for (unsigned long index = 0; index < prim->basis().size(); ++index) {
    frac_coords.block<3, 1>(0, index) = prim->basis()[index].const_frac();
  }
  xtal::Lattice lattice = prim->lattice();
  std::string title = prim->title();
  prim.reset();
  return make_prim(lattice, frac_coords, occ_dof, {}, {}, {}, title);
}

/// \brief Construct xtal::BasicStructure from poscar path
std::shared_ptr<xtal::BasicStructure const> prim_from_poscar(
    std::string &poscar_path,
    std::vector<std::vector<std::string>> const &occ_dof = {},
    double xtal_tol = TOL) {
  std::filesystem::path path(poscar_path);
  std::ifstream poscar_stream(path);
  return prim_from_poscar_stream(poscar_stream, occ_dof, xtal_tol);
}

/// \brief Construct xtal::BasicStructure from poscar string
std::shared_ptr<xtal::BasicStructure const> prim_from_poscar_str(
    std::string &poscar_str,
    std::vector<std::vector<std::string>> const &occ_dof = {},
    double xtal_tol = TOL) {
  std::istringstream poscar_stream(poscar_str);
  return prim_from_poscar_stream(poscar_stream, occ_dof, xtal_tol);
}

xtal::SimpleStructure simplestructure_from_poscar(std::string &poscar_path) {
  std::filesystem::path path(poscar_path);
  std::ifstream poscar_stream(path);
  return xtal::make_simple_structure(poscar_stream, TOL);
}

xtal::SimpleStructure simplestructure_from_poscar_str(std::string &poscar_str) {
  std::istringstream poscar_stream(poscar_str);
  return xtal::make_simple_structure(poscar_stream, TOL);
}

/// \brief Format xtal::BasicStructure as JSON string
std::string prim_to_json(
    std::shared_ptr<xtal::BasicStructure const> const &prim, bool frac,
    bool include_va) {
  jsonParser json;
  COORD_TYPE mode = frac ? FRAC : CART;
  write_prim(*prim, json, mode, include_va);
  std::stringstream ss;
  ss << json;
  return ss.str();
}

bool is_same_prim(xtal::BasicStructure const &first,
                  xtal::BasicStructure const &second) {
  return &first == &second;
}

std::shared_ptr<xtal::BasicStructure const> share_prim(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {  // for testing
  return prim;
}

std::shared_ptr<xtal::BasicStructure const> copy_prim(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {  // for testing
  return std::make_shared<xtal::BasicStructure const>(*prim);
}

xtal::Lattice const &get_prim_lattice(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  return prim->lattice();
}

Eigen::MatrixXd get_prim_coordinate_frac(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  Eigen::MatrixXd coordinate_frac(3, prim->basis().size());
  Index b = 0;
  for (auto const &site : prim->basis()) {
    coordinate_frac.col(b) = site.const_frac();
    ++b;
  }
  return coordinate_frac;
}

Eigen::MatrixXd get_prim_coordinate_cart(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  return prim->lattice().lat_column_mat() * get_prim_coordinate_frac(prim);
}

std::vector<std::vector<std::string>> get_prim_occ_dof(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  return prim->unique_names();
}

std::vector<std::vector<DoFSetBasis>> get_prim_local_dof(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  std::vector<std::vector<DoFSetBasis>> local_dof;
  Index b = 0;
  for (auto const &site : prim->basis()) {
    std::vector<DoFSetBasis> site_dof;
    for (auto const &pair : site.dofs()) {
      std::string const &dofname = pair.first;
      xtal::SiteDoFSet const &dofset = pair.second;
      site_dof.emplace_back(dofname, dofset.component_names(), dofset.basis());
    }
    local_dof.push_back(site_dof);
    ++b;
  }
  return local_dof;
}

std::vector<DoFSetBasis> get_prim_global_dof(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  std::vector<DoFSetBasis> global_dof;
  for (auto const &pair : prim->global_dofs()) {
    std::string const &dofname = pair.first;
    xtal::DoFSet const &dofset = pair.second;
    global_dof.emplace_back(dofname, dofset.component_names(), dofset.basis());
  }
  return global_dof;
}

std::map<std::string, xtal::Molecule> get_prim_molecules(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  std::map<std::string, xtal::Molecule> molecules;
  std::vector<std::vector<std::string>> mol_names = prim->unique_names();
  if (mol_names.empty()) {
    mol_names = xtal::allowed_molecule_unique_names(*prim);
  }
  Index b = 0;
  for (auto const &site_mol_names : mol_names) {
    Index i = 0;
    for (auto const &name : site_mol_names) {
      if (!molecules.count(name)) {
        molecules.emplace(name, prim->basis()[b].occupant_dof()[i]);
      }
      ++i;
    }
    ++b;
  }
  return molecules;
}

std::shared_ptr<xtal::BasicStructure const> make_within(
    std::shared_ptr<xtal::BasicStructure const> const &init_prim) {
  auto prim = std::make_shared<xtal::BasicStructure>(*init_prim);
  prim->within();
  return prim;
}

std::shared_ptr<xtal::BasicStructure const> make_primitive(
    std::shared_ptr<xtal::BasicStructure const> const &init_prim) {
  auto prim = std::make_shared<xtal::BasicStructure>(*init_prim);
  *prim = xtal::make_primitive(*prim, prim->lattice().tol());
  return prim;
}

std::shared_ptr<xtal::BasicStructure const> make_canonical_prim(
    std::shared_ptr<xtal::BasicStructure const> const &init_prim) {
  auto prim = std::make_shared<xtal::BasicStructure>(*init_prim);
  xtal::Lattice lattice{prim->lattice()};
  lattice.make_right_handed();
  lattice = xtal::canonical::equivalent(lattice);
  prim->set_lattice(xtal::canonical::equivalent(lattice), CART);
  return prim;
}

std::vector<std::vector<Index>> asymmetric_unit_indices(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  // Note: pybind11 doesn't nicely convert sets of set,
  // so return vector of vector, which is converted to list[list[int]]
  std::vector<std::vector<Index>> result;
  std::set<std::set<Index>> asym_unit = make_asymmetric_unit(*prim);
  for (auto const &orbit : asym_unit) {
    result.push_back(std::vector<Index>(orbit.begin(), orbit.end()));
  }
  return result;
}

std::vector<xtal::SymOp> make_prim_factor_group(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  return xtal::make_factor_group(*prim);
}

std::vector<xtal::SymOp> make_prim_crystal_point_group(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  auto fg = xtal::make_factor_group(*prim);
  return xtal::make_crystal_point_group(fg, prim->lattice().tol());
}

// SymOp

xtal::SymOp make_symop(Eigen::Matrix3d const &matrix,
                       Eigen::Vector3d const &translation, bool time_reversal) {
  return xtal::SymOp(matrix, translation, time_reversal);
}

std::string symop_to_json(xtal::SymOp const &op, xtal::Lattice const &lattice) {
  jsonParser json;
  to_json(op.matrix, json["matrix"]);
  to_json_array(op.translation, json["translation"]);
  to_json(op.is_time_reversal_active, json["time_reversal"]);

  std::stringstream ss;
  ss << json;
  return ss.str();
}

// SymInfo

xtal::SymInfo make_syminfo(xtal::SymOp const &op,
                           xtal::Lattice const &lattice) {
  return xtal::SymInfo(op, lattice);
}

std::string get_syminfo_type(xtal::SymInfo const &syminfo) {
  return to_string(syminfo.op_type);
}

Eigen::Vector3d get_syminfo_axis(xtal::SymInfo const &syminfo) {
  return syminfo.axis.const_cart();
}

double get_syminfo_angle(xtal::SymInfo const &syminfo) { return syminfo.angle; }

Eigen::Vector3d get_syminfo_screw_glide_shift(xtal::SymInfo const &syminfo) {
  return syminfo.screw_glide_shift.const_cart();
}

Eigen::Vector3d get_syminfo_location(xtal::SymInfo const &syminfo) {
  return syminfo.location.const_cart();
}

std::string get_syminfo_brief_cart(xtal::SymInfo const &syminfo) {
  return to_brief_unicode(syminfo, xtal::SymInfoOptions(CART));
}

std::string get_syminfo_brief_frac(xtal::SymInfo const &syminfo) {
  return to_brief_unicode(syminfo, xtal::SymInfoOptions(FRAC));
}

std::string syminfo_to_json(xtal::SymInfo const &syminfo) {
  jsonParser json;
  to_json(syminfo, json);

  to_json(to_brief_unicode(syminfo, xtal::SymInfoOptions(CART)),
          json["brief"]["CART"]);
  to_json(to_brief_unicode(syminfo, xtal::SymInfoOptions(FRAC)),
          json["brief"]["FRAC"]);

  std::stringstream ss;
  ss << json;
  return ss.str();
}

xtal::SimpleStructure make_simplestructure(
    xtal::Lattice const &lattice,
    Eigen::MatrixXd const &atom_coordinate_frac = Eigen::MatrixXd(),
    std::vector<std::string> const &atom_type = std::vector<std::string>{},
    std::map<std::string, Eigen::MatrixXd> const &atom_properties =
        std::map<std::string, Eigen::MatrixXd>{},
    Eigen::MatrixXd const &mol_coordinate_frac = Eigen::MatrixXd(),
    std::vector<std::string> const &mol_type = std::vector<std::string>{},
    std::map<std::string, Eigen::MatrixXd> const &mol_properties =
        std::map<std::string, Eigen::MatrixXd>{},
    std::map<std::string, Eigen::MatrixXd> const &global_properties =
        std::map<std::string, Eigen::MatrixXd>{}) {
  xtal::SimpleStructure simple;
  simple.lat_column_mat = lattice.lat_column_mat();
  Eigen::MatrixXd const &L = simple.lat_column_mat;
  simple.atom_info.coords = L * atom_coordinate_frac;
  simple.atom_info.names = atom_type;
  simple.atom_info.properties = atom_properties;
  simple.mol_info.coords = L * mol_coordinate_frac;
  simple.mol_info.names = mol_type;
  simple.mol_info.properties = mol_properties;
  simple.properties = global_properties;
  return simple;
}

xtal::Lattice get_simplestructure_lattice(xtal::SimpleStructure const &simple,
                                          double xtal_tol = TOL) {
  return xtal::Lattice(simple.lat_column_mat, xtal_tol);
}

Eigen::MatrixXd get_simplestructure_atom_coordinate_cart(
    xtal::SimpleStructure const &simple) {
  if (simple.atom_info.coords.cols() == 0) {
    return Eigen::MatrixXd::Zero(3, 0);
  }
  return simple.atom_info.coords;
}

Eigen::MatrixXd get_simplestructure_atom_coordinate_frac(
    xtal::SimpleStructure const &simple) {
  if (simple.atom_info.coords.cols() == 0) {
    return Eigen::MatrixXd::Zero(3, 0);
  }
  return get_simplestructure_lattice(simple).inv_lat_column_mat() *
         simple.atom_info.coords;
}

std::vector<std::string> get_simplestructure_atom_type(
    xtal::SimpleStructure const &simple) {
  return simple.atom_info.names;
}

std::map<std::string, Eigen::MatrixXd> get_simplestructure_atom_properties(
    xtal::SimpleStructure const &simple) {
  return simple.atom_info.properties;
}

Eigen::MatrixXd get_simplestructure_mol_coordinate_cart(
    xtal::SimpleStructure const &simple) {
  if (simple.mol_info.coords.cols() == 0) {
    return Eigen::MatrixXd::Zero(3, 0);
  }
  return simple.mol_info.coords;
}

Eigen::MatrixXd get_simplestructure_mol_coordinate_frac(
    xtal::SimpleStructure const &simple) {
  if (simple.mol_info.coords.cols() == 0) {
    return Eigen::MatrixXd::Zero(3, 0);
  }
  return get_simplestructure_lattice(simple).inv_lat_column_mat() *
         simple.mol_info.coords;
}

std::vector<std::string> get_simplestructure_mol_type(
    xtal::SimpleStructure const &simple) {
  return simple.mol_info.names;
}

std::map<std::string, Eigen::MatrixXd> get_simplestructure_mol_properties(
    xtal::SimpleStructure const &simple) {
  return simple.mol_info.properties;
}

std::map<std::string, Eigen::MatrixXd> get_simplestructure_global_properties(
    xtal::SimpleStructure const &simple) {
  return simple.properties;
}

xtal::SimpleStructure simplestructure_from_json(std::string const &json_str) {
  jsonParser json = jsonParser::parse(json_str);
  xtal::SimpleStructure simple;
  from_json(simple, json);
  return simple;
}

std::string simplestructure_to_json(xtal::SimpleStructure const &simple) {
  jsonParser json;
  to_json(simple, json);
  std::stringstream ss;
  ss << json;
  return ss.str();
}

std::vector<xtal::SymOp> make_simplestructure_factor_group(
    xtal::SimpleStructure const &simple) {
  std::vector<std::vector<std::string>> occ_dof;
  for (std::string name : simple.atom_info.names) {
    occ_dof.push_back({name});
  }
  std::shared_ptr<xtal::BasicStructure const> prim =
      make_prim(get_simplestructure_lattice(simple, TOL),
                get_simplestructure_atom_coordinate_frac(simple), occ_dof);
  return xtal::make_factor_group(*prim);
}

std::vector<xtal::SymOp> make_simplestructure_crystal_point_group(
    xtal::SimpleStructure const &simple) {
  auto fg = make_simplestructure_factor_group(simple);
  return xtal::make_crystal_point_group(fg, TOL);
}

xtal::SimpleStructure make_simplestructure_within(
    xtal::SimpleStructure const &init_structure) {
  xtal::SimpleStructure structure = init_structure;
  structure.within();
  return structure;
}

xtal::SimpleStructure make_superstructure(
    Eigen::Matrix3l const &transformation_matrix_to_super,
    xtal::SimpleStructure const &simple) {
  return xtal::make_superstructure(transformation_matrix_to_super.cast<int>(),
                                   simple);
}

bool simplestructure_is_equivalent_to(
    xtal::SimpleStructure const &first, xtal::SimpleStructure const &second,
    double xtal_tol = TOL,
    std::map<std::string, double> properties_tol =
        std::map<std::string, double>()) {
  return xtal::is_equivalent(first, second, xtal_tol, properties_tol);
}

std::vector<Eigen::VectorXd> make_equivalent_property_values(
    std::vector<xtal::SymOp> const &point_group, Eigen::VectorXd const &x,
    std::string property_type, Eigen::MatrixXd basis = Eigen::MatrixXd(0, 0),
    double tol = TOL) {
  AnisoValTraits traits(property_type);
  Index dim = traits.dim();
  auto compare = [&](Eigen::VectorXd const &lhs, Eigen::VectorXd const &rhs) {
    return float_lexicographical_compare(lhs, rhs, tol);
  };
  std::set<Eigen::VectorXd, decltype(compare)> equivalent_x(compare);
  if (basis.cols() == 0) {
    basis = Eigen::MatrixXd::Identity(dim, dim);
  }
  Eigen::MatrixXd basis_pinv = _xtal_impl::pseudoinverse(basis);
  for (auto const &op : point_group) {
    Eigen::VectorXd x_standard = basis * x;
    Eigen::MatrixXd M = traits.symop_to_matrix(op.matrix, op.translation,
                                               op.is_time_reversal_active);
    equivalent_x.insert(basis_pinv * M * x_standard);
  }
  return std::vector<Eigen::VectorXd>(equivalent_x.begin(), equivalent_x.end());
}

// UnitCellCoord
xtal::UnitCellCoord make_integral_site_coordinate(
    Index sublattice, Eigen::Vector3l const &unitcell) {
  return xtal::UnitCellCoord(sublattice, unitcell);
}

// UnitCellCoordRep
xtal::UnitCellCoordRep make_unitcellcoord_rep(
    xtal::SymOp const &op, xtal::BasicStructure const &prim) {
  return xtal::make_unitcellcoord_rep(op, prim.lattice(),
                                      xtal::symop_site_map(op, prim));
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(_xtal, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        libcasm.xtal
        ------------

        The libcasm.xtal module is a Python interface to the crystallography
        classes and methods in the CASM::xtal namespace of the CASM C++ libraries.
        This includes:

        - Data structures for representing lattices, crystal structures, and
          degrees of freedom (DoF).
        - Methods for enumerating lattices, making super structures,
          finding primitive and reduced cells, and finding symmetry
          operations.

    )pbdoc";

  py::class_<xtal::SimpleStructure> pyStructure(m, "Structure", R"pbdoc(
    A crystal structure

    Structure may specify atom and / or molecule coordinates and properties:

    - lattice vectors
    - atom coordinates
    - atom type names
    - continuous atom properties
    - molecule coordinates
    - molecule type names
    - continuous molecule properties
    - continuous global properties

    Atom representation is most widely supported in CASM methods. In some limited cases the molecule representation is used.

    Notes
    -----

    The positions of atoms or molecules in the crystal structure is defined by the lattice and atom coordinates or molecule coordinates. If included, strain and displacement properties, which are defined in reference to an ideal state, should be interpreted as the strain and displacement that takes the crystal from the ideal state to the state specified by the structure lattice and atom or molecule coordinates. The convention used by CASM is that displacements are applied first, and then the displaced coordinates and lattice vectors are strained.

    See the CASM `Degrees of Freedom (DoF) and Properties`_
    documentation for the full list of supported properties and their
    definitions.

    .. rubric:: Special Methods

    - Structure may be copied with `copy.copy` or `copy.deepcopy`.


    .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/

    )pbdoc");

  py::class_<xtal::Lattice>(m, "Lattice", R"pbdoc(
      A 3-dimensional lattice

      .. rubric:: Special Methods

      Sort :class:`~libcasm.xtal.Lattice` by how canonical the lattice vectors are via ``<``, ``<=``, ``>``, ``>=`` (see also :ref:`Lattice Canonical Form <lattice-canonical-form>`), and check if lattice are approximately equal via ``==``, ``!=``:

      .. code-block:: Python

          import libcasm.xtal as xtal

          L1 = xtal.Lattice( # less canonical, lesser
              np.array(
                  [
                      [0.0, 0.0, 2.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                  ]
              ).transpose()
          )
          L2 = xtal.Lattice( # more canonical, greater
              np.array(
                  [
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 2.0],
                  ]
              ).transpose()
          )
          assert L1 < L2
          assert L1 <= L2
          assert L2 > L1
          assert L2 >= L1
          assert (L1 == L2) == False
          assert L1 != L2
          assert L1 == L1
          assert (L1 != L1) == False
          assert xtal.is_equivalent_to(L1, L2) == True


      .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/
      )pbdoc")
      .def(py::init<Eigen::Matrix3d const &, double, bool>(),
           "Construct a Lattice", py::arg("column_vector_matrix"),
           py::arg("tol") = TOL, py::arg("force") = false, R"pbdoc(

      Parameters
      ----------
      column_vector_matrix : array_like, shape=(3,3)
          The lattice vectors, as columns of a 3x3 matrix.
      tol : float = :data:`~libcasm.casmglobal.TOL`
          Tolerance to be used for crystallographic comparisons.
      )pbdoc")
      .def("column_vector_matrix", &xtal::Lattice::lat_column_mat,
           "Returns the lattice vectors, as columns of a 3x3 matrix.")
      .def("tol", &xtal::Lattice::tol,
           "Returns the tolerance used for crystallographic comparisons.")
      .def("set_tol", &xtal::Lattice::set_tol, py::arg("tol"),
           "Set the tolerance used for crystallographic comparisons.")
      .def(py::self < py::self,
           "Sorts lattices by how canonical the lattice vectors are")
      .def(py::self <= py::self,
           "Sort lattices by how canonical the lattice vectors are")
      .def(py::self > py::self,
           "Sort lattices by how canonical the lattice vectors are")
      .def(py::self >= py::self,
           "Sort lattices by how canonical the lattice vectors are")
      .def(py::self == py::self,
           "True if lattice vectors are approximately equal")
      .def(py::self != py::self,
           "True if lattice vectors are not approximately equal")
      .def("is_equivalent_to", &lattice_is_equivalent_to, py::arg("other"),
           R"pbdoc(
            Check if self is equivalent to lattice2

            Two lattices, L1 and L2, are equivalent (i.e. have the same
            lattice points) if there exists U such that:

            .. code-block:: Python

                L1 = L2 @ U,

            where L1 and L2 are the lattice vectors as matrix columns, and
            U is a unimodular matrix (integer matrix, with abs(det(U))==1).

            Parameters
            ----------
            lattice2 : ~libcasm.xtal.Lattice
                The second lattice.

            Returns
            -------
            is_equivalent: bool
                True if self is equivalent to lattice2.
            )pbdoc")
      .def("is_superlattice_of", &is_superlattice_of, py::arg("lattice2"),
           R"pbdoc(
      Check if lattice1 (self) is a superlattice of lattice2

      If lattice1 (self) is a superlattice of lattice2, then

      .. code-block:: Python

          L1 = L2 @ T

      where p is the index of a point_group operation, T is an approximately
      integer tranformation matrix T, and L1 and L2 are the lattice vectors, as
      columns of a matrix, of lattice1 and lattice2, respectively.

      Parameters
      ----------
      lattice2 : ~libcasm.xtal.Lattice
          The second lattice.

      Returns
      -------
      (is_superlattice_of, T): (bool, numpy.ndarray[numpy.float64[3, 3]])
          Returns tuple with a boolean that is True if lattice1 (self) is a
          superlattice of lattice2, and the tranformation matrix T
          such that L1 = L2 @ T. Note: If is_superlattice_of==True,
          numpy.rint(T).astype(int) can be used to round array elements to
          the nearest integer.
      )pbdoc")
      .def("is_equivalent_superlattice_of", &is_equivalent_superlattice_of,
           py::arg("lattice2"),
           py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Check if lattice1 (self) is equivalent to a superlattice of lattice2

      If lattice1 (self) is equivalent to a superlattice of lattice2, then

      .. code-block:: Python

          L1 = point_group[p].matrix() @ L2 @ T

      where p is the index of a point_group operation, T is an approximately
      integer tranformation matrix T, and L1 and L2 are the lattice vectors, as
      columns of a matrix, of lattice1 and lattice2, respectively.

      Parameters
      ----------
      lattice2 : ~libcasm.xtal.Lattice
          The second lattice.
      point_group : list[:class:`~libcasm.xtal.SymOp`]
          The point group symmetry that generates equivalent lattices. Depending
          on the use case, this is often the prim crystal point group,
          :func:`~casm.xtal.make_crystal_point_group()`, or the lattice
          point group, :func:`~casm.xtal.make_point_group()`.

      Returns
      -------
      (is_equivalent_superlattice_of, T, p): (bool,
      numpy.ndarray[numpy.float64[3, 3]], int)
          Returns tuple with a boolean that is True if lattice1 (self) is
          equivalent to a superlattice of lattice2, the
          tranformation matrix T, and point group index, p, such that L1 =
          point_group[p].matrix() @ L2 @ T. Note: If
          is_equivalent_superlattice_of==True, numpy.rint(T).astype(int) can
          be used to round array elements to the nearest integer.
      )pbdoc");

  m.def("make_canonical_lattice", &make_canonical_lattice, py::arg("lattice"),
        R"pbdoc(
    Returns the canonical equivalent lattice

    Finds the canonical right-handed Niggli cell of the lattice, applying
    lattice point group operations to find the equivalent lattice in a
    standardized orientation. The canonical orientation prefers lattice
    vectors that form symmetric matrices with large positive values on the
    diagonal and small values off the diagonal. See also `Lattice Canonical
    Form`_.

    Notes
    -----
    The returned lattice is not canonical in the context of Prim supercell
    lattices, in which case the crystal point group must be used in
    determining the canonical orientation of the supercell lattice.

    .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

    Parameters
    ----------
    init_lattice : ~libcasm.xtal.Lattice
        The initial lattice.

    Returns
    -------
    lattice : ~libcasm.xtal.Lattice
        The canonical equivalent lattice, using the lattice point group.
    )pbdoc");

  m.def("make_canonical", &make_canonical_lattice, py::arg("init_lattice"),
        "Equivalent to :func:`~casm.xtal.make_canonical_lattice`");

  m.def("fractional_to_cartesian", &fractional_to_cartesian, py::arg("lattice"),
        py::arg("coordinate_frac"), R"pbdoc(
    Convert fractional coordinates to Cartesian coordinates

    The result is equal to:

    .. code-block:: Python

        lattice.column_vector_matrix() @ coordinate_frac

    Parameters
    ----------
    lattice : ~libcasm.xtal.Lattice
        The lattice.
    coordinate_frac : array_like, shape (3, n)
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors.

    Returns
    -------
    coordinate_cart : numpy.ndarray[numpy.float64[3, n]]
        Coordinates, as columns of a matrix, in Cartesian coordinates.
    )pbdoc");

  m.def("cartesian_to_fractional", &cartesian_to_fractional, py::arg("lattice"),
        py::arg("coordinate_cart"), R"pbdoc(
    Convert Cartesian coordinates to fractional coordinates

    The result is equal to:

    .. code-block:: Python

        np.linalg.pinv(lattice.column_vector_matrix()) @ coordinate_cart

    Parameters
    ----------
    lattice : ~libcasm.xtal.Lattice
        The lattice.
    coordinate_cart : array_like, shape (3, n)
        Coordinates, as columns of a matrix, in Cartesian coordinates.

    Returns
    -------
    coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors.
    )pbdoc");

  m.def("fractional_within", &fractional_within, py::arg("lattice"),
        py::arg("init_coordinate_frac"), R"pbdoc(
    Translate fractional coordinates within the lattice unit cell

    Parameters
    ----------
    lattice : ~libcasm.xtal.Lattice
        The lattice.
    init_coordinate_frac : array_like, shape (3, n)
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors.

    Returns
    -------
    coordinate_frac : numpy.ndarray[numpy.float64[3, n]]
        Coordinates, as columns of a matrix, in fractional coordinates
        with respect to the lattice vectors, translatd within the
        lattice unit cell.
    )pbdoc");

  m.def(
      "min_periodic_displacement",
      [](xtal::Lattice const &lattice, Eigen::Vector3d const &r1,
         Eigen::Vector3d const &r2, bool robust) {
        if (robust) {
          return robust_pbc_displacement_cart(lattice, r1, r2);
        } else {
          return fast_pbc_displacement_cart(lattice, r1, r2);
        }
      },
      py::arg("lattice"), py::arg("r1"), py::arg("r2"),
      py::arg("robust") = true,
      R"pbdoc(
      Return minimum length displacement (r2 - r1), accounting for periodic
      boundary conditions.

      Parameters
      ----------
      lattice : ~libcasm.xtal.Lattice
          The lattice, defining the periodic boundaries.
      r1 : array_like, shape (3, 1)
          Position, r1, in Cartesian coordinates.
      r2 : array_like, shape (3, 1)
          Position, r2, in Cartesian coordinates.
      robust : boolean, default=True
          If True, use a "robust" method which uses the lattice's Wigner-Seitz
          cell to determine the nearest image, which guarantees to find the
          minimum distance. If False, use a "fast" method, which removes integer
          multiples of lattice translations from the displacement, but may not
          result in the true minimum distance.

      Returns
      -------
      displacement: numpy.ndarray[numpy.float64[3, 1]]]
          The displacement r2 - r1, in Cartesian coordinates, with minimum length,
          accounting for periodic boundary conditions.
      )pbdoc");

  m.def("make_point_group", &make_lattice_point_group, py::arg("lattice"),
        R"pbdoc(
      Returns the lattice point group

      Parameters
      ----------
      lattice : ~libcasm.xtal.Lattice
          The lattice.

      Returns
      -------
      point_group : list[:class:`~libcasm.xtal.SymOp`]
          The set of rigid transformations that keep the origin fixed
          (i.e. have zero translation vector) and map the lattice (i.e.
          all points that are integer multiples of the lattice vectors)
          onto itself.
      )pbdoc");

  m.def("make_transformation_matrix_to_super",
        &make_transformation_matrix_to_super, py::arg("superlattice"),
        py::arg("unit_lattice"),
        R"pbdoc(
     Returns the integer transformation matrix for the superlattice relative a unit lattice.

     Parameters
     ----------
     superlattice : ~libcasm.xtal.Lattice
         The superlattice.
     unit_lattice : ~libcasm.xtal.Lattice
         The unit lattice.

     Returns
     -------
     T: numpy.ndarray[numpy.int64[3, 3]]
         Returns the integer tranformation matrix T such that S = L @ T, where S and L
         are the lattice vectors, as columns of a matrix, of the superlattice and
         unit_lattice, respectively.

     Raises
     ------
     RuntimeError:
         If superlattice is not a superlattice of unit_lattice.
     )pbdoc");

  m.def("enumerate_superlattices", &enumerate_superlattices,
        py::arg("unit_lattice"), py::arg("point_group"), py::arg("max_volume"),
        py::arg("min_volume") = Index(1), py::arg("dirs") = std::string("abc"),
        R"pbdoc(
      Enumerate symmetrically distinct superlattices

      Superlattices satify:

      .. code-block:: Python

          S = L @ T,

      where S and L are, respectively, the superlattice and unit lattice vectors as columns of
      (3x3) matrices, and T is an integer (3x3) transformation matrix.

      Superlattices S1 and S2 are symmetrically equivalent if there exists p and U such that:

      .. code-block:: Python

          S2 = p.matrix() @ S1 @ U,

      where p is an element in the point group, and U is a unimodular matrix (integer matrix, with
      abs(det(U))==1).

      Parameters
      ----------
      unit_lattice : ~libcasm.xtal.Lattice
          The unit lattice.
      point_group : list[:class:`~libcasm.xtal.SymOp`]
          The point group symmetry that determines if superlattices are equivalent. Depending on the use case, this is often the prim crystal point group, :func:`~casm.xtal.make_crystal_point_group()`, or the lattice point group, :func:`~casm.xtal.make_point_group()`.
      max_volume : int
          The maximum volume superlattice to enumerate, as a multiple of the volume of unit_lattice.
      min_volume : int, default=1
          The minimum volume superlattice to enumerate, as a multiple of the volume of unit_lattice.
      dirs : str, default="abc"
          A string indicating which lattice vectors to enumerate over. Some combination of 'a',
          'b', and 'c', where 'a' indicates the first lattice vector of the unit cell, 'b' the
          second, and 'c' the third.

      Returns
      -------
      superlattices : list[:class:`~libcasm.xtal.Lattice`]
          A list of superlattices of the unit lattice which are distinct under application of
          point_group. The resulting lattices will be in canonical form with respect to the
          point_group.
      )pbdoc");

  m.def("make_superduperlattice", &make_superduperlattice, py::arg("lattices"),
        py::arg("mode") = std::string("commensurate"),
        py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Returns the smallest lattice that is superlattice of the input lattices

      Parameters
      ----------
      lattices : list[:class:`~libcasm.xtal.Lattice`]
          List of lattices.
      mode : str, default="commensurate"
          One of:

          - "commensurate": Returns the smallest possible superlattice of all input lattices
          - "minimal_commensurate": Returns the lattice that is the smallest possible superlattice of an equivalent lattice to all input lattice
          - "fully_commensurate": Returns the lattice that is a superlattice of all equivalents of
            all input lattices
      point_group : list[casm.xtal.symop], default=[]
          Point group that generates the equivalent lattices for the the "minimal_commensurate" and
          "fully_commensurate" modes.

      Returns
      -------
      superduperlattice : ~libcasm.xtal.Lattice
          The superduperlattice
      )pbdoc");

  py::class_<xtal::AtomPosition>(m, "AtomComponent", R"pbdoc(
      An atomic component of a molecular :class:`~casm.xtal.Occupant`
      )pbdoc")
      .def(py::init(&make_atom_position), py::arg("name"),
           py::arg("coordinate"), py::arg("properties"), R"pbdoc(

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for atoms to
          be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      coordinate : array_like, shape (3,)
          Position of the atom, in Cartesian coordinates, relative
          to the basis site at which the occupant containing this
          atom is placed.
      properties : dict[str, array_like]
          Fixed properties of the atom, such as magnetic sping or
          selective dynamics flags. Keys must be the name of a
          CASM-supported property type. Values are arrays with
          dimensions matching the standard dimension of the property
          type.

          See the CASM `Degrees of Freedom (DoF) and Properties`_
          documentation for the full list of supported properties and their
          definitions.

          .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def("name", &xtal::AtomPosition::name,
           "Returns the \"chemical name\" of the atom.")
      .def("coordinate", &xtal::AtomPosition::cart, R"pbdoc(
           Returns the position of the atom

           The osition is in Cartesian coordinates, relative to the
           basis site at which the occupant containing this atom
           is placed.
           )pbdoc")
      .def("properties", &get_atom_position_properties,
           "Returns the fixed properties of the atom");

  py::class_<xtal::Molecule>(m, "Occupant", R"pbdoc(
      A site occupant, which may be a vacancy, atom, or molecule

      The Occupant class is used to represent all chemical species,
      including single atoms, vacancies, and molecules.

      )pbdoc")
      .def(py::init(&make_molecule), py::arg("name"),
           py::arg("atoms") = std::vector<xtal::AtomPosition>{},
           py::arg("is_divisible") = false,
           py::arg("properties") = std::map<std::string, Eigen::MatrixXd>{},
           R"pbdoc(

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for occupants to
          be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      atoms : list[:class:`~libcasm.xtal.AtomComponent`], optional
          The atomic components of a molecular occupant. Atoms and
          vacancies are represented with a single AtomComponent with the
          same name for the Occupant and the AtomComponent. If atoms is
          an empty list (the default value), then an atom or vacancy is
          created, based on the name parameter.
      is_divisible : bool, default=False
          If True, indicates an Occupant that may split into components
          during kinetic Monte Carlo calculations.
      properties : dict[str, array_like], default={}
          Fixed properties of the occupant, such as magnetic
          spin or selective dynamics flags. Keys must be the name of a
          CASM-supported property type. Values are arrays with
          dimensions matching the standard dimension of the property
          type.

          See the CASM `Degrees of Freedom (DoF) and Properties`_
          documentation for the full list of supported properties and their
          definitions.

          .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def("name", &xtal::Molecule::name,
           "The \"chemical name\" of the occupant")
      .def("is_divisible", &xtal::Molecule::is_divisible,
           "True if is divisible in kinetic Monte Carlo calculations")
      .def("atoms", &xtal::Molecule::atoms,
           "Returns the atomic components of the occupant")
      .def("properties", &get_molecule_properties,
           "Returns the fixed properties of the occupant")
      .def("is_vacancy", &xtal::Molecule::is_vacancy,
           "True if occupant is a vacancy.")
      .def("is_atomic", &xtal::Molecule::is_atomic,
           "True if occupant is a single isotropic atom or vacancy");

  m.def("make_vacancy", &xtal::Molecule::make_vacancy, R"pbdoc(
      Construct a Occupant object representing a vacancy

      This function is equivalent to :func:`~libcasm.xtal.Occupant("Va")`.
      )pbdoc");

  m.def("make_atom", &xtal::Molecule::make_atom, py::arg("name"), R"pbdoc(
      Construct a Occupant object representing a single isotropic atom

      This function is equivalent to :func:`~libcasm.xtal.Occupant(name)`.

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for occupants
          to be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      )pbdoc");

  py::class_<DoFSetBasis>(m, "DoFSetBasis", R"pbdoc(
      The basis for a set of degrees of freedom (DoF)

      Degrees of freedom (DoF) are continuous-valued vectors having a
      standard basis that is related to the fixed reference frame of
      the crystal. CASM supports both site DoF, which are associated
      with a particular prim basis site, and global DoF, which are
      associated with the infinite crystal. Standard DoF types are
      implemented in CASM and a traits system allows developers to
      extend CASM to include additional types of DoF.

      In many cases, the standard basis is the appropriate choice, but
      CASM also allows for a user-specified basis in terms of the
      standard basis. A user-specified basis may fully span the
      standard basis or only a subspace. This allows:

      - restricting strain to a subspace, such as only volumetric or
        only shear strains
      - restricting displacements to a subspace, such as only within
        a particular plane
      - reorienting DoF, such as to use symmetry-adapted strain order
        parameters

      See the CASM `Degrees of Freedom (DoF) and Properties`_
      documentation for the full list of supported DoF types and their
      definitions. Some examples:

      - `"disp"`: Atomic displacement
      - `"EAstrain"`: Euler-Almansi strain metric
      - `"GLstrain"`: Green-Lagrange strain metric
      - `"Hstrain"`: Hencky strain metric
      - `"Cmagspin"`: Collinear magnetic spin
      - `"SOmagspin"`: Non-collinear magnetic spin, with spin-orbit coupling

      .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def(py::init(&make_dofsetbasis), py::arg("dofname"),
           py::arg("axis_names") = std::vector<std::string>{},
           py::arg("basis") = Eigen::MatrixXd(0, 0), R"pbdoc(

      Parameters
      ----------
      dofname : str
          The type of DoF. Must be a CASM supported DoF type.
      basis : array_like, shape (m, n), default=numpy.ndarray[numpy.float64[1, 0]]
          User-specified DoF basis vectors, as columns of a matrix. The
          DoF values in this basis, `x_prim`, are related to the DoF
          values in the CASM standard basis, `x_standard`, according to
          `x_standard = basis * x_prim`. The number of rows in the basis
          matrix must match the standard dimension of the CASM DoF type.
          The number of columns must be less than or equal to the number
          of rows. The default value indicates the standard basis should
          be used.
      axis_names : list[str], default=[]
          Names for the DoF basis vectors (i.e. names for the basis matrix
          columns). Size must match number of columns in the basis matrix.
          The axis names should be appropriate for use in latex basis
          function formulas. Example, for ``dofname="disp"``:

              axis_names=["d_{1}", "d_{2}", "d_{3}"]

          The default value indicates the standard basis should be used.
      )pbdoc")
      .def("dofname", &get_dofsetbasis_dofname, "Returns the DoF type name.")
      .def("axis_names", &get_dofsetbasis_axis_names, "Returns the axis names.")
      .def("basis", &get_dofsetbasis_basis, "Returns the basis matrix.");

  // Note: Prim is intended to be `std::shared_ptr<xtal::BasicStructure const>`,
  // but Python does not handle constant-ness directly as in C++. Therefore, do
  // not add modifiers. Bound functions should still take
  // `std::shared_ptr<xtal::BasicStructure const> const &` or
  // `xtal::BasicStructure const &` arguments and return
  // `std::shared_ptr<xtal::BasicStructure const>`. Pybind11 will cast away the
  // const-ness of the returned quantity. The one exception is the method
  // `make_prim` used for the libcasm.xtal.Prim __init__ method, which it
  // appears must return `std::shared_ptr<xtal::BasicStructure>`.

  py::class_<xtal::BasicStructure, std::shared_ptr<xtal::BasicStructure>>(
      m, "Prim", R"pbdoc(
      A primitive crystal structure and allowed degrees of freedom (DoF) (the `"Prim"`)

      The Prim specifies:

      - lattice vectors
      - crystal basis sites
      - occupation DoF,
      - continuous local (site) DoF
      - continuous global DoF.

      It is usually best practice for the Prim to be an actual primitive
      cell, but it is not forced to be. The actual primitive cell will
      have a factor group with the minimum number of symmetry operations,
      which will result in more efficient methods. Some methods may have
      unexpected results when using a non-primitive Prim.

      Notes
      -----
      The Prim is not required to have the primitive equivalent cell at
      construction. The :func:`~casm.xtal.make_primitive` method may be
      used to find the primitive equivalent, and the
      :func:`~casm.xtal.make_canonical_prim` method may be used to find
      the equivalent with a Niggli cell lattice aligned in a CASM
      standard direction.
      )pbdoc")
      .def(py::init(&make_prim), py::arg("lattice"), py::arg("coordinate_frac"),
           py::arg("occ_dof"),
           py::arg("local_dof") = std::vector<std::vector<DoFSetBasis>>{},
           py::arg("global_dof") = std::vector<DoFSetBasis>{},
           py::arg("occupants") = std::map<std::string, xtal::Molecule>{},
           py::arg("title") = std::string("prim"),
           R"pbdoc(

      .. _prim-init:

      Parameters
      ----------
      lattice : ~libcasm.xtal.Lattice
          The primitive cell Lattice.
      coordinate_frac : array_like, shape (3, n)
          Basis site positions, as columns of a matrix, in fractional
          coordinates with respect to the lattice vectors.
      occ_dof : list[list[str]]
          Labels ('orientation names') of occupants allowed on each basis
          site. The value occ_dof[b] is the list of occupants allowed on
          the `b`-th basis site. The values may either be (i) the name of
          an isotropic atom (i.e. "Mg") or vacancy ("Va"), or (ii) a key
          in the occupants dictionary (i.e. "H2O", or "H2_xx"). The names
          are case sensitive, and "Va" is reserved for vacancies.
      local_dof : list[list[:class:`~libcasm.xtal.DoFSetBasis`]], default=[[]]
          Continuous DoF allowed on each basis site. No effect if empty.
          If not empty, the value local_dof[b] is a list of :class:`DoFSetBasis`
          objects describing the DoF allowed on the `b`-th basis site.
      global_dof : list[:class:`~libcasm.xtal.DoFSetBasis`], default=[]
          Global continuous DoF allowed for the entire crystal.
      occupants : dict[str,:class:`~libcasm.xtal.Occupant`], default=[]
          :class:`Occupant` allowed in the crystal. The keys are labels
          ('orientation names') used in the occ_dof parameter. This may
          include isotropic atoms, vacancies, atoms with fixed anisotropic
          properties, and molecular occupants. A seperate key and value is
          required for all species with distinct anisotropic properties
          (i.e. "H2_xy", "H2_xz", and "H2_yz" for distinct orientations,
          or "A.up", and "A.down" for distinct collinear magnetic spins,
          etc.).
      title : str, default="prim"
          A title for the prim. When the prim is used to construct a
          cluster expansion, this must consist of alphanumeric characters
          and underscores only. The first character may not be a number.
      )pbdoc")
      .def("lattice", &get_prim_lattice, "Returns the lattice")
      .def("coordinate_frac", &get_prim_coordinate_frac,
           "Returns the basis site positions, as columns of a matrix, in "
           "fractional coordinates with respect to the lattice vectors")
      .def("coordinate_cart", &get_prim_coordinate_cart,
           "Returns the basis site positions, as columns of a matrix, in "
           "Cartesian coordinates")
      .def("occ_dof", &get_prim_occ_dof,
           "Returns the labels (orientation names) of occupants allowed on "
           "each basis site")
      .def("local_dof", &get_prim_local_dof,
           "Returns the continuous DoF allowed on each basis site")
      .def(
          "global_dof", &get_prim_global_dof,
          "Returns the continuous DoF allowed for the entire crystal structure")
      .def("occupants", &get_prim_molecules,
           "Returns the :class:`Occupant` allowed in the crystal.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data, double xtal_tol) {
            jsonParser json{data};
            ParsingDictionary<AnisoValTraits> const *aniso_val_dict = nullptr;
            return std::make_shared<xtal::BasicStructure>(
                read_prim(json, xtal_tol, aniso_val_dict));
          },
          "Construct a Prim from a Python dict. The `Prim reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/BasicStructure/>`_ documents the expected "
          "format.",
          py::arg("data"), py::arg("xtal_tol") = TOL)
      .def(
          "to_dict",
          [](std::shared_ptr<xtal::BasicStructure const> const &prim, bool frac,
             bool include_va) {
            jsonParser json;
            COORD_TYPE mode = frac ? FRAC : CART;
            write_prim(*prim, json, mode, include_va);
            return static_cast<nlohmann::json>(json);
          },
          py::arg("frac") = true, py::arg("include_va") = false,
          R"pbdoc(
            Represent the Prim as a Python dict

            Parameters
            ----------
            frac : boolean, default=True
                If True, write basis site positions in fractional coordinates
                relative to the lattice vectors. If False, write basis site positions
                in Cartesian coordinates.
            include_va : boolean, default=False
                If a basis site only allows vacancies, it is not printed by default.
                If this is True, basis sites with only vacancies will be included.

            Returns
            -------
            data : dict
                The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/>`_ documents the expected format.

            )pbdoc")
      .def_static(
          "from_json", &prim_from_json,
          "Construct a Prim from a JSON-formatted string. The `Prim reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/BasicStructure/>`_ documents the expected JSON "
          "format.",
          py::arg("prim_json_str"), py::arg("xtal_tol") = TOL)
      .def_static("from_poscar", &prim_from_poscar,
                  R"pbdoc(
            Construct a Prim from a VASP POSCAR file

            Parameters
            ----------
            poscar_path : str
                Path to the POSCAR file

            occ_dof : list[list[str]] = []
                By default, the occupation degrees of freedom (DoF) are
                set to only allow the POSCAR atom types. This may be
                provided, to explicitly set the occupation DoF.

            xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
                Tolerance used for lattice.

            Returns
            -------
            prim : ~libcasm.xtal.Prim
                A Prim

            )pbdoc",
                  py::arg("poscar_path"),
                  py::arg("occ_dof") = std::vector<std::vector<std::string>>{},
                  py::arg("xtal_tol") = TOL)
      .def_static("from_poscar_str", &prim_from_poscar_str,
                  R"pbdoc(
            Construct a Prim from a VASP POSCAR string

            Parameters
            ----------
            poscar_str : str
                VASP POSCAR as a string

            occ_dof : list[list[str]] = []
                By default, the occupation degrees of freedom (DoF) are
                set to only allow the POSCAR atom types. This may be
                provided, to explicitly set the occupation DoF.

            xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
                Tolerance used for lattice.

            Returns
            -------
            prim : ~libcasm.xtal.Prim
                A Prim

            )pbdoc",
                  py::arg("poscar_str"),
                  py::arg("occ_dof") = std::vector<std::vector<std::string>>{},
                  py::arg("xtal_tol") = TOL)
      .def_static(
          "from_atom_coordinates",
          [](xtal::SimpleStructure const &simple,
             std::vector<std::vector<std::string>> occ_dof, double xtal_tol) {
            if (occ_dof.size() == 0) {
              for (std::string name : simple.atom_info.names) {
                occ_dof.push_back({name});
              }
            }
            return make_prim(get_simplestructure_lattice(simple, xtal_tol),
                             get_simplestructure_atom_coordinate_frac(simple),
                             occ_dof);
          },
          py::arg("structure"),
          py::arg("occ_dof") = std::vector<std::vector<std::string>>{},
          py::arg("xtal_tol") = TOL, R"pbdoc(
          Construct a Prim from a Structure, using atom coordinates

          Parameters
          ----------
          structure : ~libcasm.xtal.Structure
               The input structure.

          occ_dof : list[list[str]] = []
              By default, the occupation degrees of freedom (DoF) are
              set to only allow the structure atom types. This may be
              provided, to explicitly set the occupation DoF.

          xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
              Tolerance used for the Prim lattice.

          Returns
          -------
          prim : ~libcasm.xtal.Prim
                A Prim
          )pbdoc")
      .def("to_json", &prim_to_json, py::arg("frac") = true,
           py::arg("include_va") = false,
           R"pbdoc(
            Represent the Prim as a JSON-formatted string.

            Parameters
            ----------
            frac : boolean, default=True
                If True, write basis site positions in fractional coordinates
                relative to the lattice vectors. If False, write basis site positions
                in Cartesian coordinates.
            include_va : boolean, default=False
                If a basis site only allows vacancies, it is not printed by default.
                If this is True, basis sites with only vacancies will be included.

            Returns
            -------
            data : dict
                The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/>`_ documents the expected JSON format.

            )pbdoc");

  m.def("_is_same_prim", &is_same_prim, py::arg("first"), py::arg("second"),
        R"pbdoc(
            Check if Prim are sharing the same data

            This is for testing purposes, it should be equivalent to
            `first is second` and `first == second`.

            Parameters
            ----------
            first : ~libcasm.xtal.Prim
                First Prim.

            second : ~libcasm.xtal.Prim
                Second Prim.

            Returns
            -------
            is_same : ~libcasm.xtal.Prim
                Returns true if Prim are sharing the same data

            )pbdoc");

  m.def("_share_prim", &share_prim, py::arg("init_prim"), R"pbdoc(
            Make a copy of a Prim - sharing same data

            This is for testing purposes.

            Parameters
            ----------
            init_prim : ~libcasm.xtal.Prim
                Initial prim.

            Returns
            -------
            prim : ~libcasm.xtal.Prim
                A copy of the initial prim, sharing the same data.

            )pbdoc");

  m.def("_copy_prim", &copy_prim, py::arg("init_prim"), R"pbdoc(
            Make a copy of a Prim - not sharing same data

            This is for testing purposes.

            Parameters
            ----------
            init_prim : ~libcasm.xtal.Prim
                Initial prim.

            Returns
            -------
            prim : ~libcasm.xtal.Prim
                A copy of the initial prim, not sharing the same data.

            )pbdoc");

  m.def("make_prim_within", &make_within, py::arg("init_prim"), R"pbdoc(
            Returns an equivalent Prim with all basis site coordinates within the unit cell

            Parameters
            ----------
            init_prim : ~libcasm.xtal.Prim
                The initial prim.

            Returns
            -------
            prim : ~libcasm.xtal.Prim
                The prim with all basis site coordinates within the unit cell.

            )pbdoc");

  m.def("make_within", &make_within, py::arg("init_prim"),
        "Equivalent to :func:`~casm.xtal.make_prim_within`");

  m.def("make_primitive", &make_primitive, py::arg("init_prim"), R"pbdoc(
            Returns a primitive equivalent Prim

            A :class:`Prim` object is not forced to be the primitive equivalent
            cell at construction. This function finds and returns the primitive
            equivalent cell by checking for internal translations that map all
            basis sites onto equivalent basis sites, including allowed
            occupants and equivalent local degrees of freedom (DoF), if they
            exist.

            Parameters
            ----------
            init_prim : ~libcasm.xtal.Prim
                The initial prim.

            Returns
            -------
            prim : ~libcasm.xtal.Lattice
                The primitive equivalent prim.
            )pbdoc");

  m.def("make_canonical_prim", &make_canonical_prim, py::arg("init_prim"),
        R"pbdoc(
          Returns an equivalent Prim with canonical lattice

          Finds the canonical right-handed Niggli cell of the lattice, applying
          lattice point group operations to find the equivalent lattice in a
          standardized orientation. The canonical orientation prefers lattice
          vectors that form symmetric matrices with large positive values on the
          diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

          .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

          Parameters
          ----------
          init_prim : ~libcasm.xtal.Prim
              The initial prim.

          Returns
          -------
          prim : ~libcasm.xtal.Lattice
              The prim with canonical lattice.

        )pbdoc");

  m.def("make_canonical", &make_canonical_prim, py::arg("init_prim"),
        "Equivalent to :func:`~casm.xtal.make_canonical_prim`");

  m.def("asymmetric_unit_indices", &asymmetric_unit_indices, py::arg("prim"),
        R"pbdoc(
          Returns the indices of equivalent basis sites

          Parameters
          ----------
          prim : ~libcasm.xtal.Prim
              The prim.

          Returns
          -------
          asymmetric_unit_indices : list[list[int]]
              One list of basis site indices for each set of symmetrically equivalent basis sites.
              In other words, the elements of asymmetric_unit_indices[i] are the indices of the
              i-th set of basis sites which are symmetrically equivalent to each other.

          )pbdoc");

  m.def("make_prim_factor_group", &make_prim_factor_group, py::arg("prim"),
        R"pbdoc(
          Returns the factor group

          Parameters
          ----------
          prim : ~libcasm.xtal.Prim
              The prim.

          Returns
          -------
          factor_group : list[:class:`~libcasm.xtal.SymOp`]
              The the set of symmery operations, with translation lying within the primitive unit
              cell, that leave the lattice vectors, basis site coordinates, and all DoF invariant.

          )pbdoc");

  m.def("make_factor_group", &make_prim_factor_group, py::arg("prim"),
        "Equivalent to :func:`~casm.xtal.make_prim_factor_group`");

  m.def("make_prim_crystal_point_group", &make_prim_crystal_point_group,
        py::arg("prim"),
        R"pbdoc(
          Returns the crystal point group

          Parameters
          ----------
          prim : ~libcasm.xtal.Prim
              The prim.

          Returns
          -------
          crystal_point_group : list[:class:`~libcasm.xtal.SymOp`]
              The crystal point group is the group constructed from the prim factor group operations
              with translation vector set to zero.

          )pbdoc");

  m.def("make_crystal_point_group", &make_prim_crystal_point_group,
        py::arg("prim"),
        "Equivalent to :func:`~casm.xtal.make_prim_crystal_point_group`");

  py::class_<xtal::SymOp>(m, "SymOp", R"pbdoc(
      A symmetry operation representation that acts on Cartesian coordinates

      A SymOp, op, transforms a Cartesian coordinate according to:

      .. code-block:: Python

          r_after = op.matrix() @ r_before + op.translation()

      where r_before and r_after are shape=(3,) arrays with the Cartesian
      coordinates before and after transformation, respectively.

      Additionally, the sign of magnetic spins is flipped according to:

      .. code-block:: Python

          if op.time_reversal() is True:
              s_after = -s_before

      where s_before and s_after are the spins before and after
      transformation, respectively.

      .. rubric:: Special Methods

      The multiplication operator ``X = lhs * rhs`` can be used to apply SymOp to various objects:

      - ``X=SymOp``, ``lhs=SymOp``, ``rhs=SymOp``: Construct the :class:`~libcasm.xtal.SymOp`, `X`, equivalent to applying first `rhs`, then `lhs`.
      - ``X=np.ndarray``, ``lhs=SymOp``, ``rhs=np.ndarray``: Transform multiple Cartesian coordinates, represented as columns of a `np.ndarray`.
      - ``X=dict[str,np.ndarray]``, ``lhs=SymOp``, ``rhs=dict[str,np.ndarray]``: Transform CASM-supported properties (local or global). Keys must be the name of a CASM-supported property type. Values are arrays with the number of rows matching the standard dimension of the property type. For local properties, columns correspond to the value associated with each site. For global properties, there is one column. See the CASM `Degrees of Freedom (DoF) and Properties`_ documentation for the full list of supported properties and their definitions.
      - ``X=Lattice``, ``lhs=SymOp``, ``rhs=Lattice``: Transform a :class:`~libcasm.xtal.Lattice`.
      - ``X=Structure``, ``lhs=SymOp``, ``rhs=Structure``: Transform a :class:`~libcasm.xtal.Structure`.

      .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/

      )pbdoc")
      .def(py::init(&make_symop), py::arg("matrix"), py::arg("translation"),
           py::arg("time_reversal"),
           R"pbdoc(

          Parameters
          ----------
          matrix : array_like, shape (3, 3)
              The transformation matrix component of the symmetry operation.
          translation : array_like, shape (3,)
              Translation component of the symmetry operation.
          time_reversal : bool
              True if the symmetry operation includes time reversal (spin flip),
              False otherwise
          )pbdoc")
      .def("matrix", &xtal::get_matrix,
           "Returns the transformation matrix value.")
      .def("translation", &xtal::get_translation,
           "Returns the translation value.")
      .def("time_reversal", &xtal::get_time_reversal,
           "Returns the time reversal value.")
      .def(
          "__mul__",
          [](xtal::SymOp const &op, Eigen::Vector3d const &coordinate_cart) {
            return get_matrix(op) * coordinate_cart + get_translation(op);
          },
          py::arg("coordinate_cart"),
          "Transform Cartesian coordinates, represented as a 1d array")
      .def(
          "__mul__",
          [](xtal::SymOp const &op, Eigen::MatrixXd const &coordinate_cart) {
            Eigen::MatrixXd transformed = get_matrix(op) * coordinate_cart;
            for (Index i = 0; i < transformed.cols(); ++i) {
              transformed.col(i) += get_translation(op);
            }
            return transformed;
          },
          py::arg("coordinate_cart"),
          "Transform multiple Cartesian coordinates, represented as columns of "
          "a matrix.")
      .def(
          "__mul__",
          [](xtal::SymOp const &lhs, xtal::SymOp const &rhs) {
            return lhs * rhs;
          },
          py::arg("rhs"),
          "Construct the SymOp equivalent to applying first rhs, then this.")
      .def(
          "__mul__",
          [](xtal::SymOp const &op,
             std::map<std::string, Eigen::MatrixXd> const &properties) {
            return copy_apply(op, properties);
          },
          py::arg("rhs"),
          "Transform CASM-supported properties (local or global).")
      .def(
          "__mul__",
          [](xtal::SymOp const &op, xtal::Lattice const &lattice) {
            return sym::copy_apply(op, lattice);
          },
          py::arg("lattice"), "Transform a Lattice.")
      .def(
          "__mul__",
          [](xtal::SymOp const &op, xtal::SimpleStructure const &simple) {
            return copy_apply(op, simple);
          },
          py::arg("structure"), "Transform a Structure.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};
            Eigen::Matrix3d matrix;
            from_json(matrix, json["matrix"]);
            Eigen::Vector3d translation;
            from_json(translation, json["tau"]);
            bool time_reversal;
            from_json(time_reversal, json["time_reversal"]);
            return xtal::SymOp(matrix, translation, time_reversal);
          },
          "Construct a SymOp from a Python dict. The `Coordinate "
          "Transformation Representation reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "symmetry/SymGroup/"
          "#coordinate-transformation-representation-json-object>`_ documents "
          "the expected format.",
          py::arg("data"))
      .def(
          "to_dict",
          [](xtal::SymOp const &op) {
            jsonParser json;
            json["matrix"] = xtal::get_matrix(op);
            to_json_array(xtal::get_translation(op), json["tau"]);
            json["time_reversal"] = xtal::get_time_reversal(op);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the SymOp as a Python dict. The `Coordinate "
          "Transformation Representation reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "symmetry/SymGroup/"
          "#coordinate-transformation-representation-json-object>`_ documents "
          "the format.");

  py::class_<xtal::SymInfo>(m, "SymInfo", R"pbdoc(
      Symmetry operation type, axis, invariant point, etc.

      )pbdoc")
      .def(py::init<xtal::SymOp const &, xtal::Lattice const &>(),
           py::arg("op"), py::arg("lattice"),
           R"pbdoc(

          Parameters
          ----------
          op : ~libcasm.xtal.SymOp
              The symmetry operation.
          lattice : ~libcasm.xtal.Lattice
              The lattice
          )pbdoc")
      .def("op_type", &get_syminfo_type, R"pbdoc(
          Returns the symmetry operation type.

          Returns
          -------
          op_type: str
              One of:

              - "identity"
              - "mirror"
              - "glide"
              - "rotation"
              - "screw"
              - "inversion"
              - "rotoinversion"
              - "invalid"
          )pbdoc")
      .def("axis", get_syminfo_axis, R"pbdoc(
          Returns the symmetry operation axis.

          Returns
          -------
          axis: numpy.ndarray[numpy.float64[3, 1]]
              This is:

              - the rotation axis, if the operation is a rotation or screw operation
              - the rotation axis of inversion * self, if this is an improper rotation (then the axis is a normal vector for a mirror plane)
              - zero vector, if the operation is identity or inversion

              The axis is in Cartesian coordinates and normalized to length 1.
          )pbdoc")
      .def("angle", &get_syminfo_angle, R"pbdoc(
          Returns the symmetry operation angle.

          Returns
          -------
          angle: float
              This is:

              - the rotation angle, if the operation is a rotation or screw operation
              - the rotation angle of inversion * self, if this is an improper rotation (then the axis is a normal vector for a mirror plane)
              - zero, if the operation is identity or inversion

          )pbdoc")
      .def("screw_glide_shift", &get_syminfo_screw_glide_shift, R"pbdoc(
          Returns the screw or glide translation component

          Returns
          -------
          screw_glide_shift: numpy.ndarray[numpy.float64[3, 1]]
              This is:

              - the component of translation parallel to `axis`, if the
                operation is a rotation
              - the component of translation perpendicular to `axis`, if
                the operation is a mirror

              The screw_glide_shift is in Cartesian coordinates.
          )pbdoc")
      .def("location", &get_syminfo_location, R"pbdoc(
          A Cartesian coordinate that is invariant to the operation (if one exists)

          Returns
          -------
          location: numpy.ndarray[numpy.float64[3, 1]]
              The location is in Cartesian coordinates. This does not exist for the identity
              operation.
          )pbdoc")
      .def("brief_cart", &get_syminfo_brief_cart, R"pbdoc(
          A brief description of the symmetry operation, in Cartesian coordinates

          Returns
          -------
          brief_cart: str
              A brief string description of the symmetry operation, in Cartesian coordinates,
              following the conventions of (International Tables for Crystallography (2015). Vol.
              A. ch. 1.4, pp. 50-59).
          )pbdoc")
      .def("brief_frac", &get_syminfo_brief_frac, R"pbdoc(
          A brief description of the symmetry operation, in fractional coordinates

          Returns
          -------
          brief_cart: str
              A brief string description of the symmetry operation, in fractional coordinates,
              following the conventions of (International Tables for Crystallography (2015). Vol.
              A. ch. 1.4, pp. 50-59).
          )pbdoc")
      .def(
          "to_dict",
          [](xtal::SymInfo const &syminfo) {
            jsonParser json;
            to_json(syminfo, json);

            to_json(to_brief_unicode(syminfo, xtal::SymInfoOptions(CART)),
                    json["brief"]["CART"]);
            to_json(to_brief_unicode(syminfo, xtal::SymInfoOptions(FRAC)),
                    json["brief"]["FRAC"]);
            return static_cast<nlohmann::json>(json);
          },
          "Represent SymInfo as a Python dict. The `Symmetry Operation "
          "Information reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "symmetry/SymGroup/#symmetry-operation-json-object/>`_ documents the "
          "format.")
      .def("to_json", &syminfo_to_json, R"pbdoc(
          Represent the symmetry operation information as a JSON-formatted string.

          The `Symmetry Operation Information JSON Object reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/symmetry/SymGroup/#symmetry-operation-json-object/>`_ documents JSON format, except conjugacy class and inverse operation are not currently included.
          )pbdoc");

  pyStructure
      .def(
          py::init(&make_simplestructure), py::arg("lattice"),
          py::arg("atom_coordinate_frac") = Eigen::MatrixXd(),
          py::arg("atom_type") = std::vector<std::string>{},
          py::arg("atom_properties") = std::map<std::string, Eigen::MatrixXd>{},
          py::arg("mol_coordinate_frac") = Eigen::MatrixXd(),
          py::arg("mol_type") = std::vector<std::string>{},
          py::arg("mol_properties") = std::map<std::string, Eigen::MatrixXd>{},
          py::arg("global_properties") =
              std::map<std::string, Eigen::MatrixXd>{},
          R"pbdoc(

    Parameters
    ----------
    lattice : ~libcasm.xtal.Lattice
        The Lattice. Note: The lattice tolerance is not saved in Structure.
    atom_coordinate_frac : array_like, shape (3, n)
        Atom positions, as columns of a matrix, in fractional
        coordinates with respect to the lattice vectors.
    atom_type : list[str], size=n
        Atom type names.
    atom_properties : dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with individual atoms, if present. Keys must be the name of a CASM-supported property type. Values are arrays with dimensions matching the standard dimension of the property type.
    mol_coordinate_frac : array_like, shape (3, n)
        Molecule positions, as columns of a matrix, in fractional
        coordinates with respect to the lattice vectors.
    mol_type : list[str], size=n
        Molecule type names.
    mol_properties : dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with individual molecules, if present. Keys must be the name of a CASM-supported property type. Values are arrays with dimensions matching the standard dimension of the property type.
    global_properties : dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with entire crystal, if present. Keys must be the name of a CASM-supported property type. Values are (m, 1) arrays with dimensions matching the standard dimension of the property type.
    )pbdoc")
      .def("lattice", &get_simplestructure_lattice, R"pbdoc(
            Returns the lattice

            Parameters
            ----------
            xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
                Tolerance used for lattice.

            Returns
            -------
            lattice : ~libcasm.xtal.Lattice
                The lattice

            )pbdoc",
           py::arg("xtal_tol") = TOL)
      .def("atom_coordinate_cart", &get_simplestructure_atom_coordinate_cart,
           "Returns the atom positions, as columns of a matrix, in Cartesian "
           "coordinates.")
      .def("atom_coordinate_frac", &get_simplestructure_atom_coordinate_frac,
           "Returns the atom positions, as columns of a matrix, in fractional "
           "coordinates with respect to the lattice vectors.")
      .def("atom_type", &get_simplestructure_atom_type,
           "Returns a list with atom type names.")
      .def("atom_properties", &get_simplestructure_atom_properties,
           "Returns continuous properties associated with individual atoms, if "
           "present.")
      .def("mol_coordinate_cart", &get_simplestructure_mol_coordinate_cart,
           "Returns the molecule positions, as columns of a matrix, in "
           "Cartesian coordinates.")
      .def("mol_coordinate_frac", &get_simplestructure_mol_coordinate_frac,
           "Returns the molecule positions, as columns of a matrix, in "
           "fractional coordinates with respect to the lattice vectors.")
      .def("mol_type", &get_simplestructure_mol_type,
           "Returns a list with molecule type names.")
      .def(
          "mol_properties", &get_simplestructure_mol_properties,
          "Returns continuous properties associated with individual molecules, "
          "if present.")
      .def("global_properties", &get_simplestructure_global_properties,
           "Returns continuous properties associated with the entire crystal, "
           "if present.")
      .def_static(
          "from_dict",
          [](const nlohmann::json &data) {
            jsonParser json{data};
            std::cout << "JSON: " << json << std::endl;
            xtal::SimpleStructure simple;
            from_json(simple, json);
            return simple;
          },
          "Construct a Structure from a Python dict. The `Structure reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/SimpleStructure/>`_ documents the expected "
          "format.",
          py::arg("data"))
      .def(
          "to_dict",
          [](xtal::SimpleStructure const &simple) {
            jsonParser json;
            to_json(simple, json);
            return static_cast<nlohmann::json>(json);
          },
          "Represent the Structure as a Python dict. The `Structure reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/SimpleStructure/>`_ documents the format.")
      .def_static(
          "from_json", &simplestructure_from_json,
          "Construct a Structure from a JSON-formatted string. The `Structure "
          "reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/SimpleStructure/>`_ documents the expected JSON "
          "format.",
          py::arg("structure_json_str"))
      .def_static("from_poscar", &simplestructure_from_poscar,
                  R"pbdoc(
            Construct a Structure from a VASP POSCAR file

            Parameters
            ----------
            poscar_path : str
                Path to the POSCAR file

            Returns
            -------
            struture : ~libcasm.xtal.Structure
                A Structure

            )pbdoc",
                  py::arg("poscar_path"))
      .def_static("from_poscar_str", &simplestructure_from_poscar_str,
                  R"pbdoc(
            Construct a Structure from a VASP POSCAR string

            Parameters
            ----------
            poscar_str : str
                The POSCAR as a string

            Returns
            -------
            struture : ~libcasm.xtal.Structure
                A Structure

            )pbdoc",
                  py::arg("poscar_str"))
      .def("to_json", &simplestructure_to_json,
           "Represent the Structure as a JSON-formatted string. The `Structure "
           "reference "
           "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
           "crystallography/SimpleStructure/>`_ documents the expected JSON "
           "format.")
      .def(
          "to_poscar_str",
          [](xtal::SimpleStructure const &structure, bool sort,
             std::string title, std::vector<std::string> ignore,
             bool cart_coordinate_mode) -> std::string {
            VaspIO::PrintPOSCAR p{structure, title};
            p.ignore() = {};
            for (auto const &atom_type : ignore) {
              p.ignore().insert(atom_type);
            }
            if (sort) {
              p.sort();
            }
            if (cart_coordinate_mode) {
              p.set_cart();
            }
            std::stringstream ss;
            p.print(ss);
            return ss.str();
          },
          R"pbdoc(
            Convert a Structure to a VASP POSCAR string

            Parameters
            ----------
            sort: bool = True
                If True, sort atoms by atom type name. Otherwise, print in the
                order appearing in the structure.

            title: str
                The POSCAR title line

            ignore: list[str] = ["Va", "VA", "va"]
                Atom names to ignore and not include in the POSCAR. By default,
                vacancies are not included. To include vacancies, use `ignore=[]`.

            cart_coordinate_mode: bool = False
                If True, write POSCAR using Cartesian coordinates.

            Returns
            -------
            struture : ~libcasm.xtal.Structure
                A Structure

            )pbdoc",
          py::arg("sort") = true, py::arg("title") = "<title>",
          py::arg("ignore") = std::vector<std::string>{"VA", "Va", "va"},
          py::arg("cart_coordinate_mode") = false)
      .def("__copy__",
           [](xtal::SimpleStructure const &self) {
             return xtal::SimpleStructure(self);
           })
      .def("__deepcopy__", [](xtal::SimpleStructure const &self,
                              py::dict) { return xtal::SimpleStructure(self); })
      .def("is_equivalent_to", &simplestructure_is_equivalent_to,
           py::arg("structure2"), py::arg("xtal_tol") = TOL,
           py::arg("properties_tol") = std::map<std::string, double>(), R"pbdoc(
              Check if self is equivalent to structure2

              Notes
              -----

              Two structures are equivalent if they have:

              - equivalent lattices (i.e. have the same lattice points,
                up to the specified tolerance)
              - equivalent atoms and molecules, including:

                - equivalent coordinates, accounting for periodic boundary
                  conditions, up to the specified tolerance
                - identical names
                - equal site properties, up to the specified tolerance

              - equal global properties, up to the specified tolerance

              This method does not check for rotations or translations that are
              not integer multiples of the lattice vectors. For structures that
              are equivalent after a rotation, or after translation of basis
              sites, this returns false. That type of equivalence should be checked
              using the methods in libcasm-mapping.

              Parameters
              ----------
              structure2 : ~libcasm.xtal.Structure
                  The second structure.
              xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
                  Tolerance used for lattice and coordinate comparisons.
              properties_tol: dict[str,float] = {}
                  Tolerance used for properties comparisons, by global or local
                  property name. If a property name is not present, "default"
                  will be used. If "default" is not present, the default CASM
                  tolerance (:data:`~libcasm.casmglobal.TOL`) will be used.

              Returns
              -------
              is_equivalent: bool
                  True if self is equivalent to structure2.
              )pbdoc");

  m.def("make_structure_factor_group", &make_simplestructure_factor_group,
        py::arg("structure"), R"pbdoc(
           Returns the factor group of an atomic structure

           Parameters
           ----------
           structure : ~libcasm.xtal.Structure
               The structure.

           Returns
           -------
           factor_group : list[:class:`~libcasm.xtal.SymOp`]
               The the set of symmery operations, with translation lying within the primitive unit
               cell, that leave the lattice vectors, atom coordinates, and atom types invariant.

           Notes
           -----
           Currently this method only considers atom coordinates and types. Molecular coordinates
           and types are not considered. Properties are not considered. The default CASM tolerance
           is used for comparisons. To consider molecules or properties, or to use a different
           tolerance, use a Prim.

           )pbdoc");

  m.def("make_factor_group", &make_simplestructure_factor_group,
        py::arg("structure"),
        "Equivalent to :func:`~casm.xtal.make_structure_factor_group`");

  m.def("make_structure_crystal_point_group",
        &make_simplestructure_crystal_point_group, py::arg("structure"),
        R"pbdoc(
           Returns the crystal point group of an atomic structure

           Parameters
           ----------
           structure : ~libcasm.xtal.Structure
               The structure.

           Returns
           -------
           crystal_point_group : list[:class:`~libcasm.xtal.SymOp`]
               The crystal point group is the group constructed from the structure factor group
               operations with translation vector set to zero.

           Notes
           -----
           Currently this method only considers atom coordinates and types. Molecular coordinates
           and types are not considered. Properties are not considered. The default CASM tolerance
           is used for comparisons. To consider molecules or properties, or to use a different
           tolerance, use a Prim.
           )pbdoc");

  m.def("make_crystal_point_group", &make_simplestructure_crystal_point_group,
        py::arg("structure"),
        "Equivalent to :func:`~casm.xtal.make_structure_crystal_point_group`");

  m.def("make_structure_within", &make_simplestructure_within,
        py::arg("init_structure"), R"pbdoc(
            Returns an equivalent Structure with all atom and mol site coordinates within the unit cell

            Parameters
            ----------
            init_structure : ~libcasm.xtal.Structure
                The initial structure.

            Returns
            -------
            structure : ~libcasm.xtal.Structure
                The structure with all atom and mol site coordinates within the unit cell.

            )pbdoc");

  m.def("make_within", &make_simplestructure_within, py::arg("init_structure"),
        "Equivalent to :func:`~casm.xtal.make_structure_within`");

  m.def("make_superstructure", &make_superstructure,
        py::arg("transformation_matrix_to_super"), py::arg("structure"),
        R"pbdoc(
      Make a superstructure

      Parameters
      ----------
      transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
          The transformation matrix, T, relating the superstructure lattice vectors, S, to the unit structure lattice vectors, L, according to S = L @ T, where S and L are shape=(3,3)  matrices with lattice vectors as columns.
      structure: ~libcasm.xtal.Structure
          The unit structure used to form the superstructure.

      Returns
      -------
      superstructure: ~libcasm.xtal.Structure
          The superstructure.
      )pbdoc");

  m.def("make_equivalent_property_values", &make_equivalent_property_values,
        py::arg("point_group"), py::arg("x"), py::arg("property_type"),
        py::arg("basis") = Eigen::MatrixXd(0, 0), py::arg("tol") = TOL,
        R"pbdoc(
      Make the set of symmetry equivalent property values

      Parameters
      ----------
      point_group : list[:class:`~libcasm.xtal.Symop`]
          Point group that generates the equivalent property values.
      x : array_like, shape=(m,1)
          The property value, as a vector. For strain, this is the
          unrolled strain metric vector. For local property values, such
          as atomic displacements, this is the vector value associated
          with one site.
      property_type : string
          The property type name. See the CASM `Degrees of Freedom (DoF) and Properties`_
          documentation for the full list of supported properties and their
          definitions.

          .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      basis : array_like, shape=(s,m), optional
          The basis in which the value is expressed, as columns of a
          matrix. A property value in this basis, `x`, is related to a
          property value in the CASM standard basis, `x_standard`,
          according to `x_standard = basis @ x`. The number of rows in
          the basis matrix must match the standard dimension of the CASM
          supported property_type. The number of columns must be less
          than or equal to the number of rows. The default value indicates
          the standard basis should be used.
      tol: float, default=1e-5
          The tolerance used to eliminate equivalent property values


      Returns
      -------
      equivalent_x: list[numpy.ndarray[numpy.float64[m, 1]]]
          A list of distinct property values, in the given basis,
          equivalent under the point group.
      )pbdoc");

  py::class_<xtal::StrainConverter>(m, "StrainConverter", R"pbdoc(
    Convert strain values

    Converts between strain metric vector values
    (6-element or less vector representing a symmetric strain metric), and
    the strain metric matrix values, or the deformation tensor, F, shape=(3,3).

    For more information on strain metrics and using a symmetry-adapted or user-specified basis, see :ref:`Strain DoF <sec-strain-dof>`.

    :class:`~casm.xtal.StrainConverter` supports the following choices of symmetric strain metrics, :math:`E`, shape=(3,3):

    - `"GLstrain"`: Green-Lagrange strain metric, :math:`E = \frac{1}{2}(F^{\mathsf{T}} F - I)`
    - `"Hstrain"`: Hencky strain metric, :math:`E = \frac{1}{2}\ln(F^{\mathsf{T}} F)`
    - `"EAstrain"`: Euler-Almansi strain metric, :math:`E = \frac{1}{2}(I−(F F^{\mathsf{T}})^{-1})`
    - `"Ustrain"`: Right stretch tensor, :math:`E = U`
    - `"Bstrain"`: Biot strain metric, :math:`E = U - I`

    )pbdoc")
      .def(py::init<std::string, Eigen::MatrixXd const &>(),
           py::arg("metric") = "Ustrain",
           py::arg("basis") = Eigen::MatrixXd::Identity(6, 6),
           R"pbdoc(

    Parameters
    ----------
    metric: str (optional, default='Ustrain')
        Choice of strain metric, one of: 'Ustrain', 'GLstrain', 'Hstrain', 'EAstrain', 'Bstrain'

    basis: array-like of shape (6, dim), optional
        User-specified basis for E_vector, in terms of the standard basis.

            E_vector_in_standard_basis = basis @ E_vector

        The default value, shape=(6,6) identity matrix, chooses the standard basis.

    )pbdoc")
      .def("metric", &xtal::StrainConverter::metric,
           "Returns the strain metric name.")
      .def("basis", &xtal::StrainConverter::basis,
           R"pbdoc(
          Returns the basis used for strain metric vectors.

          Returns
          -------
          basis: array-like of shape (6, dim), optional
              The basis for E_vector, in terms of the standard basis.

                  E_vector_in_standard_basis = basis @ E_vector

          )pbdoc")
      .def("dim", &xtal::StrainConverter::dim,
           R"pbdoc(
          Returns the strain space dimension.

          Returns
          -------
          dim: int
              The strain space dimension, equivalent to the number of columns
              of the basis matrix.
          )pbdoc")
      .def("basis_pinv", &xtal::StrainConverter::basis_pinv,
           R"pbdoc(
          Returns the strain metric basis pseudoinverse.

          Returns
          -------
          basis_pinv: numpy.ndarray[numpy.float64[dim, 6]]
              The pseudoinverse of the basis for E_vector.

                  E_vector = basis_pinv @ E_vector_in_standard_basis

          )pbdoc")
      .def_static("F_to_QU", &xtal::StrainConverter::F_to_QU, py::arg("F"),
                  R"pbdoc(
           Decompose a deformation tensor as QU.

           Parameters
           ----------
           F: numpy.ndarray[numpy.float64[3, 3]]
               The deformation tensor, :math:`F`.

           Returns
           -------
           Q:
               The shape=(3,3) isometry matrix, :math:`Q`, of the
               deformation tensor.
           U:
               The shape=(3,3) right stretch tensor, :math:`U`, of
               the deformation tensor.
           )pbdoc")
      .def_static("F_to_VQ", &xtal::StrainConverter::F_to_VQ, py::arg("F"),
                  R"pbdoc(
            Decompose a deformation tensor as VQ.

            Parameters
            ----------
            F: numpy.ndarray[numpy.float64[3, 3]]
                The deformation tensor, :math:`F`.

            Returns
            -------
            Q:
                The shape=(3,3) isometry matrix, :math:`Q`, of the
                deformation tensor.
            V:
                The shape=(3,3) left stretch tensor, :math:`V`, of
                the deformation tensor.
            )pbdoc")
      .def("to_F", &xtal::StrainConverter::to_F, py::arg("E_vector"),
           R"pbdoc(
           Convert strain metric vector to deformation tensor.

           Parameters
           ----------
           E_vector: array_like, shape=(dim,1)
               Strain metric vector, expressed in the basis of this StrainConverter.

           Returns
           -------
           F: numpy.ndarray[numpy.float64[3, 3]]
               The deformation tensor, :math:`F`.
           )pbdoc")
      .def("from_F", &xtal::StrainConverter::from_F, py::arg("F"),
           R"pbdoc(
           Convert deformation tensor to strain metric vector.

           Parameters
           ----------
           F: numpy.ndarray[numpy.float64[3, 3]]
               The deformation tensor, :math:`F`.

           Returns
           -------
           E_vector: array_like, shape=(dim,1)
               Strain metric vector, expressed in the basis of this StrainConverter.
           )pbdoc")
      .def("to_standard_basis", &xtal::StrainConverter::to_standard_basis,
           py::arg("E_vector"),
           R"pbdoc(
           Convert strain metric vector to standard basis

           Parameters
           ----------
           E_vector: array_like, shape=(dim,1)
               Strain metric vector, expressed in the basis of this StrainConverter.

           Returns
           -------
           E_vector_in_standard_basis: array_like, shape=(6,1)
               Strain metric vector, expressed in the standard basis. This is
               equivalent to `basis @ E_vector`.
           )pbdoc")
      .def("from_standard_basis", &xtal::StrainConverter::from_standard_basis,
           py::arg("E_vector_in_standard_basis"),
           R"pbdoc(
           Convert strain metric vector from standard basis to converter basis.

           Parameters
           ----------
           E_vector_in_standard_basis: array_like, shape=(dim,1)
               Strain metric vector, expressed in the standard basis. This is
               equivalent to `basis @ E_vector`.

           Returns
           -------
           E_vector: array_like, shape=(dim,1)
               Strain metric vector, expressed in the basis of this StrainConverter.
           )pbdoc")
      .def("to_E_matrix", &xtal::StrainConverter::to_E_matrix,
           py::arg("E_vector"),
           R"pbdoc(
           Convert strain metric vector to strain metric matrix.

           Parameters
           ----------
           E_vector: array_like, shape=(dim,1)
               Strain metric vector, expressed in the basis of this StrainConverter.

           Returns
           -------
           E_matrix: array_like, shape=(3,3)
               Strain metric matrix, :math:`E`, using the metric of this StrainConverter.
           )pbdoc")
      .def("from_E_matrix", &xtal::StrainConverter::from_E_matrix,
           py::arg("E_matrix"),
           R"pbdoc(
           Convert strain metric matrix to strain metric vector.

           Parameters
           ----------
           E_matrix: array_like, shape=(3,3)
               Strain metric matrix, :math:`E`, using the metric of this StrainConverter.

           Returns
           -------
           E_vector: array_like, shape=(dim,1)
               Strain metric vector, expressed in the basis of this StrainConverter.
           )pbdoc");

  m.def("make_symmetry_adapted_strain_basis",
        &xtal::make_symmetry_adapted_strain_basis,
        R"pbdoc(
      Returns the symmetry-adapted strain basis.

      The symmetry-adapted strain basis,

      .. math::

          B^{\vec{e}} = \left(
            \begin{array}{cccccc}
            1/\sqrt{3} & 1/\sqrt{2} & -1/\sqrt{6} & 0 & 0 & 0 \\
            1/\sqrt{3} & -1/\sqrt{2} & -1/\sqrt{6} & 0 & 0 & 0  \\
            1/\sqrt{3} & 0 & 2/\sqrt{6} & 0 & 0 & 0  \\
            0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1
            \end{array}
          \right),

      which decomposes strain space into irreducible subspaces (subspaces which do not mix under application of symmetry).

      For more information on strain metrics and the symmetry-adapted strain basis, see :ref:`Strain DoF <sec-strain-dof>`.

      Returns
      -------
      symmetry_adapted_strain_basis: list[numpy.ndarray[numpy.float64[6, 6]]]
          The symmetry-adapted strain basis, :math:`B^{\vec{e}}`.
      )pbdoc");

  // IntegralSiteCoordinate -- declaration
  py::class_<xtal::UnitCellCoord> pyIntegralSiteCoordinate(
      m, "IntegralSiteCoordinate", R"pbdoc(
      Specify a site using integer sublattice and unit cell indices

      .. rubric:: Special Methods

      Translate an :class:`~libcasm.xtal.IntegralSiteCoordinate` using operators ``+``, ``-``, ``+=``, ``-=``:

      .. code-block:: Python

          import numpy as np
          from libcasm.xtal import IntegralSiteCoordinate

          # construct IntegralSiteCoordinate
          b = 0
          unitcell = np.array([1, 2, 3])
          translation = np.array([0, 0, 1])
          integral_site_coordinate = IntegralSiteCoordinate(b, unitcell)

          # translate via `+=`:
          integral_site_coordinate += translation

          # translate via `-=`:
          integral_site_coordinate -= translation

          # copy & translate via `+`:
          translated_integral_site_coordinate = integral_site_coordinate + translation

          # copy & translate via `-`:
          translated_integral_site_coordinate = integral_site_coordinate - translation


      Sort :class:`~libcasm.xtal.IntegralSiteCoordinate` by lexicographical order of unit cell indices `[i, j, k]` then sublattice index `b` using ``<``, ``<=``, ``>``, ``>=``, and compare using ``==``, ``!=``:

      .. code-block:: Python

          import numpy as np
          from libcasm.xtal import IntegralSiteCoordinate

          # construct IntegralSiteCoordinate
          b = 0
          unitcell = np.array([1, 2, 3])
          translation = np.array([0, 0, 1])
          A = IntegralSiteCoordinate(0, np.array([1, 2, 3]))
          B = IntegralSiteCoordinate(1, np.array([1, 2, 3]))

          assert A < B
          assert A <= B
          assert A <= A
          assert B > A
          assert B >= A
          assert B >= B
          assert A == A
          assert B == B
          assert A != B

      Represent :class:`~libcasm.xtal.IntegralSiteCoordinate` as the string ``"b, i j k"``, where `b` is the sublattice index and `i j k` are the unit cell indices, using ``str()``:

      .. code-block:: Python

          import numpy as np
          from libcasm.xtal import IntegralSiteCoordinate

          # construct IntegralSiteCoordinate
          site = IntegralSiteCoordinate(0, np.array([1, 2, 3]))

          assert str(site) == "0, 1 2 3"

      )pbdoc");

  // IntegralSiteCoordinateRep -- declaration
  py::class_<xtal::UnitCellCoordRep> pyIntegralSiteCoordinateRep(
      m, "IntegralSiteCoordinateRep", R"pbdoc(
      Symmetry representation for transforming IntegralSiteCoordinate

      .. rubric:: Special Methods

      Transform an :class:`~libcasm.xtal.IntegralSiteCoordinate` via multiplication operator ``*``:

      .. code-block:: Python

          from libcasm.xtal import IntegralSiteCoordinate, IntegralSiteCoordinateRep
          rep = IntegralSiteCoordinateRep(...)
          integral_site_coordinate = IntegralSiteCoordinate(...)
          transformed_integral_site_coordinate = rep * integral_site_coordinate

      )pbdoc");

  // IntegralSiteCoordinate -- definition
  pyIntegralSiteCoordinate
      .def(py::init(&make_integral_site_coordinate),
           "Construct an IntegralSiteCoordinate", py::arg("sublattice"),
           py::arg("unitcell"), R"pbdoc(

      Parameters
      ----------
      sublattice : int
          Specify a sublattice in a prim, in range [0, prim.basis().size()).
      unitcell : array_like of int, shape=(3,)
          Specify a unit cell, as multiples of the prim lattice vectors.
      )pbdoc")
      .def_static(
          "from_coordinate_cart",
          [](Eigen::Vector3d const &coordinate_cart,
             xtal::BasicStructure const &prim, double tol) {
            return xtal::UnitCellCoord::from_coordinate(
                prim,
                xtal::Coordinate(coordinate_cart, prim.lattice(), CASM::CART),
                tol);
          },
          py::arg("coordinate_cart"), py::arg("prim"),
          py::arg("tol") = CASM::TOL,
          "Construct an integral site coordinate with given Cartesian "
          "coordinate with respect to a particular Prim. An exception is "
          "raised if not possible to the given tolerance.")
      .def_static(
          "from_coordinate_frac",
          [](Eigen::Vector3d const &coordinate_frac,
             xtal::BasicStructure const &prim, double tol) {
            return xtal::UnitCellCoord::from_coordinate(
                prim,
                xtal::Coordinate(coordinate_frac, prim.lattice(), CASM::FRAC),
                tol);
          },
          py::arg("coordinate_frac"), py::arg("prim"),
          py::arg("tol") = CASM::TOL,
          "Construct an integral site coordinate with given fractional "
          "coordinate with respect to a particular Prim. An exception is "
          "raised if not possible to the given tolerance.")
      .def("sublattice", &xtal::UnitCellCoord::sublattice,
           "Returns the sublattice index.")
      .def(
          "unitcell",
          [](xtal::UnitCellCoord const &self) {
            return static_cast<Eigen::Vector3l>(self.unitcell());
          },
          "Returns the unit cell indices.")
      .def(
          "__str__",
          [](xtal::UnitCellCoord const &self) {
            std::stringstream ss;
            ss << self;
            return ss.str();
          },
          "Represent IntegralSiteCoordinate as `b, i j k`, where `b` is the "
          "sublattice index and `i j k` are the unit cell coordinates.")
      .def(
          "to_list",
          [](xtal::UnitCellCoord const &self) {
            std::vector<Index> list;
            for (int i = 0; i < 4; ++i) {
              list.push_back(self[i]);
            }
            return list;
          },
          "Represent IntegralSiteCoordinate as `[b, i, j, k]`.")
      .def_static(
          "from_list",
          [](std::vector<int> const &list) {
            if (list.size() != 4) {
              throw std::runtime_error(
                  "Error constructing IntegralSiteCoordinate from a list: size "
                  "!= 4");
            }
            return xtal::UnitCellCoord(list[0], list[1], list[2], list[3]);
          },
          "Construct IntegralSiteCoordinate from a list `[b, i, j, k]`.")
      .def(
          "__iadd__",
          [](xtal::UnitCellCoord &self, Eigen::Vector3l const &translation) {
            self += xtal::UnitCell(translation);
            return self;
          },
          py::arg("translation"),
          "Translates the integral site coordinate by adding unit cell indices")
      .def(
          "__add__",
          [](xtal::UnitCellCoord const &self,
             Eigen::Vector3l const &translation) {
            return self + xtal::UnitCell(translation);
          },
          py::arg("translation"),
          "Translates the integral site coordinate by adding unit cell indices")
      .def(
          "__isub__",
          [](xtal::UnitCellCoord &self, Eigen::Vector3l const &translation) {
            self -= xtal::UnitCell(translation);
            return self;
          },
          py::arg("translation"),
          "Translates the integral site coordinate by subtracting unit cell "
          "indices")
      .def(
          "__sub__",
          [](xtal::UnitCellCoord const &self,
             Eigen::Vector3l const &translation) {
            return self - xtal::UnitCell(translation);
          },
          py::arg("translation"),
          "Translates the integral site coordinate by subtracting unit cell "
          "indices")
      .def(
          "coordinate_cart",
          [](xtal::UnitCellCoord const &self,
             xtal::BasicStructure const &prim) {
            return self.coordinate(prim).const_cart();
          },
          py::arg("prim"),
          "Return the Cartesian coordinate corresponding to this integral site "
          "coordinate in the given Prim")
      .def(
          "coordinate_frac",
          [](xtal::UnitCellCoord const &self,
             xtal::BasicStructure const &prim) {
            return self.coordinate(prim).const_frac();
          },
          py::arg("prim"),
          "Return the fractional coordinate corresponding to this integral "
          "site coordinate in the given Prim")
      .def(py::self < py::self,
           "Sorts coordinates by lexicographical order of [i, j, k] then b")
      .def(py::self <= py::self,
           "Sorts coordinates by lexicographical order of [i, j, k] then b")
      .def(py::self > py::self,
           "Sorts coordinates by lexicographical order of [i, j, k] then b")
      .def(py::self >= py::self,
           "Sorts coordinates by lexicographical order of [i, j, k] then b")
      .def(py::self == py::self, "True if coordinates are equal")
      .def(py::self != py::self, "True if coordinates are not equal");

  // IntegralSiteCoordinateRep -- definition
  pyIntegralSiteCoordinateRep
      .def(py::init(&CASMpy::make_unitcellcoord_rep),
           "Construct an IntegralSiteCoordinateRep", py::arg("op"),
           py::arg("prim"), R"pbdoc(

      Parameters
      ----------
      op : ~libcasm.xtal.SymOp
          The symmetry operation.
      prim : ~libcasm.xtal.Prim
          The prim defining IntegralSiteCoordinate that will be transformed.
      )pbdoc")
      .def(
          "__mul__",
          [](xtal::UnitCellCoordRep const &rep,
             xtal::UnitCellCoord const &integral_site_coordinate) {
            return copy_apply(rep, integral_site_coordinate);
          },
          py::arg("integral_site_coordinate"),
          "Transform an :class:`~libcasm.xtal.IntegralSiteCoordinate`");

  m.def(
      "apply",
      [](xtal::UnitCellCoordRep const &rep,
         xtal::UnitCellCoord &integral_site_coordinate) {
        return apply(rep, integral_site_coordinate);
      },
      py::arg("rep"), py::arg("integral_site_coordinate"),
      "Applies the symmetry operation represented by the `rep` to "
      "transform `integral_site_coordinate`.");

  m.def(
      "copy_apply",
      [](xtal::UnitCellCoordRep const &rep,
         xtal::UnitCellCoord const &integral_site_coordinate) {
        return copy_apply(rep, integral_site_coordinate);
      },
      py::arg("rep"), py::arg("integral_site_coordinate"),
      "Creates a copy of `integral_site_coordinate` and applies the symmetry "
      "operation represented by `rep`.");

  m.def(
      "pretty_json",
      [](const nlohmann::json &data) -> std::string {
        jsonParser json{data};
        std::stringstream ss;
        ss << json << std::endl;
        return ss.str();
      },
      "Pretty-print JSON to string.", py::arg("data"));

  // SiteIndexConverter
  py::class_<xtal::UnitCellCoordIndexConverter>(m, "SiteIndexConverter",
                                                R"pbdoc(
      Convert between integral site indices :math:`(b,i,j,k)` and linear site index :math:`l`.
      )pbdoc")
      .def(py::init<Eigen::Matrix3l const &, int>(),
           py::arg("transformation_matrix_to_super"), py::arg("n_sublattice"),
           R"pbdoc(

          Parameters
          ----------
          transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
              The transformation matrix, T, relating the superstructure lattice vectors, S, to the unit structure lattice vectors, L, according to S = L @ T, where S and L are shape=(3,3)  matrices with lattice vectors as columns.

          n_sublattice: int
              The number of sublattices in the :class:`~libcasm.xtal.Prim`.

          )pbdoc")
      .def("never_bring_within",
           &xtal::UnitCellCoordIndexConverter::never_bring_within,
           R"pbdoc(
            Prevent the index converter from bringing :class:`~libcasm.xtal.IntegralSiteCoordinate` within the supercell when querying for the index.
          )pbdoc")
      .def("always_bring_within",
           &xtal::UnitCellCoordIndexConverter::always_bring_within,
           R"pbdoc(
            Automatically bring :class:`~libcasm.xtal.IntegralSiteCoordinate` values within the supercell when querying for the index (on by default).
          )pbdoc")
      .def("bring_within", &xtal::UnitCellCoordIndexConverter::bring_within,
           R"pbdoc(
          Bring the given :class:`~libcasm.xtal.IntegralSiteCoordinate` into the superlattice using superlattice translations.
          )pbdoc",
           py::arg("integral_site_coordinate"))
      .def(
          "linear_site_index",
          [](xtal::UnitCellCoordIndexConverter const &f,
             xtal::UnitCellCoord const &bijk) -> Index { return f(bijk); },
          R"pbdoc(
           Given the :class:`~libcasm.xtal.IntegralSiteCoordinate`, retreive its corresponding linear index. By default, if :func:`~libcasm.xtal.SiteIndexConverter.never_bring_within` has not been called, the :class:`~libcasm.xtal.IntegralSiteCoordinate` is brought within the superlattice using superlattice translations.
           )pbdoc",
          py::arg("integral_site_coordinate"))
      .def(
          "integral_site_coordinate",
          [](xtal::UnitCellCoordIndexConverter const &f,
             Index const &linear_site_index) -> xtal::UnitCellCoord {
            return f(linear_site_index);
          },
          R"pbdoc(
           Given the linear index, retreive the corresponding :class:`~libcasm.xtal.IntegralSiteCoordinate`.
           )pbdoc",
          py::arg("linear_site_index"))
      .def("total_sites", &xtal::UnitCellCoordIndexConverter::total_sites,
           R"pbdoc(
           Returns the total number of sites within the superlattice.
           )pbdoc");

  // UnitCellIndexConverter
  py::class_<xtal::UnitCellIndexConverter>(m, "UnitCellIndexConverter", R"pbdoc(
      Convert between unit cell indices :math:`(i,j,k)` and linear unit cell index.

      For each supercell, CASM generates an ordering of lattice sites :math:`(i,j,k)`.
      )pbdoc")
      .def(py::init<Eigen::Matrix3l const &>(),
           py::arg("transformation_matrix_to_super"),
           R"pbdoc(

          Parameters
          ----------
          transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
              The transformation matrix, T, relating the superstructure lattice vectors, S, to the unit structure lattice vectors, L, according to S = L @ T, where S and L are shape=(3,3)  matrices with lattice vectors as columns.

          )pbdoc")
      .def(
          "never_bring_within",
          // &xtal::UnitCellIndexConverter::never_bring_within,
          [](xtal::UnitCellIndexConverter &f) { f.never_bring_within(); },
          R"pbdoc(
            Prevent the index converter from bringing unit cell indices :math:`(i,j,k)` within the supercell when querying for the index.
          )pbdoc")
      .def(
          "always_bring_within",
          // &xtal::UnitCellIndexConverter::always_bring_within,
          [](xtal::UnitCellIndexConverter &f) { f.always_bring_within(); },
          R"pbdoc(
            Automatically bring unit cell indices :math:`(i,j,k)` within the supercell when querying for the index (on by default).
          )pbdoc")
      .def(
          "bring_within",
          //&xtal::UnitCellIndexConverter::bring_within,
          [](xtal::UnitCellIndexConverter &f, Eigen::Vector3l const &unitcell) {
            return f.bring_within(unitcell);
          },
          R"pbdoc(
           Bring the given :class:`~libcasm.xtal.IntegralSiteCoordinate` into the superlattice using superlattice translations.
           )pbdoc",
          py::arg("unitcell"))
      .def(
          "linear_unitcell_index",
          [](xtal::UnitCellIndexConverter const &f,
             Eigen::Vector3l const &unitcell) -> Index { return f(unitcell); },
          R"pbdoc(
           Given unitcell indices, :math:`(i,j,k)`, retreive the corresponding linear unitcell index. By default, if :func:`~libcasm.xtal.IntegralSiteCoordinateConverter.never_bring_within` has not been called, the lattice point is brought within the superlattice using superlattice translations.
           )pbdoc",
          py::arg("unitcell"))
      .def(
          "unitcell",
          [](xtal::UnitCellIndexConverter const &f,
             Index const &linear_unitcell_index) -> Eigen::Vector3l {
            return f(linear_unitcell_index);
          },
          R"pbdoc(
           Given the linear unitcell index, retreive the corresponding unitcell indices :math:`(i,j,k)`.
           )pbdoc",
          py::arg("linear_unitcell_index"))
      .def(
          "total_unitcells",
          //&xtal::UnitCellIndexConverter::total_sites,
          [](xtal::UnitCellIndexConverter const &f) { return f.total_sites(); },
          R"pbdoc(
           Returns the total number of unitcells within the superlattice.
           )pbdoc")
      .def(
          "make_lattice_points",
          [](xtal::UnitCellIndexConverter const &f) {
            std::vector<Eigen::Vector3l> lattice_points;
            for (Index i = 0; i < f.total_sites(); ++i) {
              lattice_points.push_back(f(i));
            }
            return lattice_points;
          },
          R"pbdoc(
           Returns a list of unitcell indices, :math:`(i,j,k)`, in the superlattice.
           )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
