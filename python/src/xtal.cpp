#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>

#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/BasicStructureTools.hh"
#include "casm/crystallography/CanonicalForm.hh"
#include "casm/crystallography/LatticeIsEquivalent.hh"
#include "casm/crystallography/SimpleStructure.hh"
#include "casm/crystallography/SimpleStructureTools.hh"
#include "casm/crystallography/Strain.hh"
#include "casm/crystallography/SuperlatticeEnumerator.hh"
#include "casm/crystallography/SymInfo.hh"
#include "casm/crystallography/SymTools.hh"
#include "casm/crystallography/io/BasicStructureIO.hh"
#include "casm/crystallography/io/SimpleStructureIO.hh"
#include "casm/crystallography/io/SymInfo_json_io.hh"
#include "casm/crystallography/io/SymInfo_stream_io.hh"

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
  jsonParser json{prim_json_str};
  ParsingDictionary<AnisoValTraits> const *aniso_val_dict = nullptr;
  return std::make_shared<xtal::BasicStructure>(
      read_prim(json, xtal_tol, aniso_val_dict));
}

/// \brief Construct xtal::BasicStructure from poscar path
std::shared_ptr<xtal::BasicStructure const> prim_from_poscar(std::string &poscar_path){
    std::filesystem::path path(poscar_path);
    std::ifstream poscar_stream(path);
    return std::make_shared<xtal::BasicStructure>(
        xtal::BasicStructure::from_poscar_stream(poscar_stream));
    
}

/// \brief Format xtal::BasicStructure as JSON string
std::string prim_to_json(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  jsonParser json;
  write_prim(*prim, json, FRAC);
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
  return xtal::allowed_molecule_names(*prim);
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
  // so return vector of vector, which is converted to List[List[int]]
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

xtal::Lattice get_simplestructure_lattice(xtal::SimpleStructure const &simple) {
  return xtal::Lattice(simple.lat_column_mat);
}

Eigen::MatrixXd get_simplestructure_atom_coordinate_cart(
    xtal::SimpleStructure const &simple) {
  return simple.atom_info.coords;
}

Eigen::MatrixXd get_simplestructure_atom_coordinate_frac(
    xtal::SimpleStructure const &simple) {
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
  return simple.mol_info.coords;
}

Eigen::MatrixXd get_simplestructure_mol_coordinate_frac(
    xtal::SimpleStructure const &simple) {
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

xtal::SimpleStructure simplestructure_from_json(jsonParser const &json) {
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
      make_prim(get_simplestructure_lattice(simple),
                get_simplestructure_atom_coordinate_frac(simple), occ_dof);
  return xtal::make_factor_group(*prim);
}

std::vector<xtal::SymOp> make_simplestructure_crystal_point_group(
    xtal::SimpleStructure const &simple) {
  auto fg = make_simplestructure_factor_group(simple);
  return xtal::make_crystal_point_group(fg, TOL);
}

xtal::SimpleStructure make_superstructure(
    Eigen::Matrix3l const &transformation_matrix_to_super,
    xtal::SimpleStructure const &simple) {
  return xtal::make_superstructure(transformation_matrix_to_super.cast<int>(),
                                   simple);
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

/// \brief Holds strain metric and basis to facilitate conversions
struct StrainConverter {
  StrainConverter(std::string _metric, Eigen::MatrixXd const &_basis)
      : metric(_metric),
        basis(_basis),
        basis_pinv(_xtal_impl::pseudoinverse(basis)) {}

  /// \brief Name of strain metric (i.e. 'Hstrain', etc.)
  std::string metric;

  /// \brief Strain metric basis, such that E_standard = basis * E_basis
  Eigen::MatrixXd basis;

  /// \brief Pseudoinverse of basis
  Eigen::MatrixXd basis_pinv;
};

/// \brief Decompose deformation tensor, F, as Q*U
///
/// \param F A deformation tensor
/// \returns {Q, U}
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> F_to_QU(Eigen::Matrix3d const &F) {
  Eigen::Matrix3d right_stretch = polar_decomposition(F);
  Eigen::Matrix3d isometry = F * right_stretch.inverse();
  return std::make_pair(isometry, right_stretch);
}

/// \brief Decompose deformation tensor, F, as V*Q
///
/// \param F A deformation tensor
/// \returns {Q, V} Note the isometry matrix, Q, is returned first.
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> F_to_VQ(Eigen::Matrix3d const &F) {
  auto result = F_to_QU(F);
  result.second = F * result.first.transpose();
  return result;
}

/// \brief Returns strain metric vector value in standard basis
Eigen::VectorXd strain_metric_vector_in_standard_basis(
    StrainConverter const &converter, Eigen::VectorXd const &E_vector) {
  return converter.basis * E_vector;
}

/// \brief Returns strain metric vector value in converter basis
Eigen::VectorXd strain_metric_vector_in_converter_basis(
    StrainConverter const &converter,
    Eigen::VectorXd const &E_vector_in_standard_basis) {
  return converter.basis_pinv * E_vector_in_standard_basis;
}

/// \brief Converter strain metric vector value to matrix value
///
/// Strain metric vector value is:
///   [Exx, Eyy, Ezz, sqrt(2)*Eyz, sqrt(2)*Exz, sqrt(2)*Exy]
///
/// \param converter Strain converter
/// \param E_vector, strain metric vector value in basis converter.basis
/// \returns E_matrix Strain metric matrix
///
Eigen::Matrix3d strain_metric_vector_to_matrix(
    StrainConverter const &converter, Eigen::VectorXd const &E_vector) {
  Eigen::VectorXd e = converter.basis * E_vector;
  double w = sqrt(2.);
  Eigen::Matrix3d E_matrix;
  E_matrix <<  //
      e(0),
      e(5) / w, e(4) / w,        //
      e(5) / w, e(1), e(3) / w,  //
      e(4) / w, e(3) / w, e(2);  //
  return E_matrix;
}

/// \brief Converter strain metric matrix value to vector value
///
/// Strain metric vector value is:
///   [Exx, Eyy, Ezz, sqrt(2)*Eyz, sqrt(2)*Exz, sqrt(2)*Exy]
///
/// \param converter Strain converter
/// \param E_matrix Strain metric matrix
/// \return E_vector, strain metric vector value in converter.basis
///
Eigen::VectorXd strain_metric_matrix_to_vector(
    StrainConverter const &converter, Eigen::Matrix3d const &E_matrix) {
  Eigen::Matrix3d const &e = E_matrix;
  Eigen::VectorXd E_vector = Eigen::VectorXd::Zero(6);
  double w = std::sqrt(2.);
  E_vector << e(0, 0), e(1, 1), e(2, 2), w * e(1, 2), w * e(0, 2), w * e(0, 1);
  return converter.basis_pinv * E_vector;
}

/// \brief Converter strain metric value to deformation tensor
///
/// \param converter A StrainConverter
/// \param E_vector Unrolled strain metric value, in converter.basis,
///     such that E_standard = converter.basis * E_vector
/// \returns F, the deformation tensor
Eigen::Matrix3d strain_metric_vector_to_F(StrainConverter const &converter,
                                          Eigen::VectorXd const &E_vector) {
  using namespace strain;
  Eigen::Matrix3d E_matrix =
      strain_metric_vector_to_matrix(converter, E_vector);

  if (converter.metric == "Hstrain") {
    return metric_to_deformation_tensor<METRIC::HENCKY>(E_matrix);
  } else if (converter.metric == "EAstrain") {
    return metric_to_deformation_tensor<METRIC::EULER_ALMANSI>(E_matrix);
  } else if (converter.metric == "GLstrain") {
    return metric_to_deformation_tensor<METRIC::GREEN_LAGRANGE>(E_matrix);
  } else if (converter.metric == "Bstrain") {
    return metric_to_deformation_tensor<METRIC::BIOT>(E_matrix);
  } else if (converter.metric == "Ustrain") {
    return E_matrix;
  } else {
    std::stringstream ss;
    ss << "StrainConverter error: Unexpected metric: " << converter.metric;
    throw std::runtime_error(ss.str());
  }
};

/// \brief Converter strain metric value to deformation tensor
///
/// \param converter A StrainConverter
/// \param F Deformation gradient tensor
/// \returns Unrolled strain metric value, in converter.basis,
///     such that E_standard = converter.basis * E_vector
Eigen::VectorXd strain_metric_vector_from_F(StrainConverter const &converter,
                                            Eigen::Matrix3d const &F) {
  using namespace strain;
  Eigen::Matrix3d E_matrix;

  if (converter.metric == "Hstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::HENCKY>(F);
  } else if (converter.metric == "EAstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::EULER_ALMANSI>(F);
  } else if (converter.metric == "GLstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::GREEN_LAGRANGE>(F);
  } else if (converter.metric == "Bstrain") {
    E_matrix = deformation_tensor_to_metric<METRIC::BIOT>(F);
  } else if (converter.metric == "Ustrain") {
    E_matrix = right_stretch_tensor(F);
  } else {
    std::stringstream ss;
    ss << "StrainConverter error: Unexpected metric: " << converter.metric;
    throw std::runtime_error(ss.str());
  }
  return strain_metric_matrix_to_vector(converter, E_matrix);
};

Eigen::MatrixXd make_symmetry_adapted_strain_basis() {
  Eigen::MatrixXd B;
  // clang-format off
  B <<  //
      1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3), 0, 0, 0,    //e1
      1 / sqrt(2), -1 / sqrt(2), 0.0, 0, 0, 0,           //e2
      -1 / sqrt(6), -1 / sqrt(6), 2 / sqrt(6), 0, 0, 0,  //e3
      0, 0, 0, 1, 0, 0,                                  //e4
      0, 0, 0, 0, 1, 0,                                  //e5
      0, 0, 0, 0, 0, 1;                                  //e6
  // clang-format on
  return B.transpose();
}

}  // namespace CASMpy

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

PYBIND11_MODULE(xtal, m) {
  using namespace CASMpy;

  m.doc() = R"pbdoc(
        casm.xtal
        ---------

        The casm.xtal module is a Python interface to the crystallography
        classes and methods in the CASM::xtal namespace of the CASM C++ libraries.
        This includes:

        - Data structures for representing lattices, crystal structures, and
          degrees of freedom (DoF).
        - Methods for enumerating lattices, making super structures,
          finding primitive and reduced cells, and finding symmetry
          operations.

    )pbdoc";

  m.attr("TOL") = TOL;

  py::class_<xtal::Lattice>(m, "Lattice", R"pbdoc(
      A 3-dimensional lattice
      )pbdoc")
      .def(py::init<Eigen::Matrix3d const &, double, bool>(),
           "Construct a Lattice", py::arg("column_vector_matrix"),
           py::arg("tol") = TOL, py::arg("force") = false, R"pbdoc(

      Parameters
      ----------
      column_vector_matrix : array_like, shape=(3,3)
          The lattice vectors, as columns of a 3x3 matrix.
      tol : float, default=xtal.TOL
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
           "True if lattice vectors are not approximately equal");

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
    init_lattice : casm.xtal.Lattice
        The initial lattice.

    Returns
    -------
    lattice : casm.xtal.Lattice
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
    lattice : casm.xtal.Lattice
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
    lattice : casm.xtal.Lattice
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
    lattice : casm.xtal.Lattice
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

  m.def("make_point_group", &make_lattice_point_group, py::arg("lattice"),
        R"pbdoc(
      Returns the lattice point group

      Parameters
      ----------
      lattice : casm.xtal.Lattice
          The lattice.

      Returns
      -------
      point_group : List[casm.xtal.SymOp]
          The set of rigid transformations that keep the origin fixed
          (i.e. have zero translation vector) and map the lattice (i.e.
          all points that are integer multiples of the lattice vectors)
          onto itself.
      )pbdoc");

  m.def("is_equivalent_to", &xtal::is_equivalent, py::arg("lattice1"),
        py::arg("lattice2"), R"pbdoc(
      Check if lattice1 is equivalent to lattice2

      Two lattices, L1 and L2, are equivalent (i.e. have the same
      lattice points) if there exists U such that:

      .. code-block:: Python

          L1 = L2 @ U,

      where L1 and L2 are the lattice vectors as matrix columns, and
      U is a unimodular matrix (integer matrix, with abs(det(U))==1).

      Parameters
      ----------
      lattice1 : casm.xtal.Lattice
          The first lattice.
      lattice2 : casm.xtal.Lattice
          The second lattice.

      Returns
      -------
      is_equivalent: bool
          True if lattice1 is equivalent to lattice2.
      )pbdoc");

  m.def("is_superlattice_of", &is_superlattice_of, py::arg("lattice1"),
        py::arg("lattice2"), R"pbdoc(
      Check if lattice1 is a superlattice of lattice2

      If lattice1 is a superlattice of lattice2, then

      .. code-block:: Python

          L1 = L2 @ T

      where p is the index of a point_group operation, T is an approximately
      integer tranformation matrix T, and L1 and L2 are the lattice vectors, as
      columns of a matrix, of lattice1 and lattice2, respectively.

      Parameters
      ----------
      lattice1 : casm.xtal.Lattice
          The first lattice.
      lattice2 : casm.xtal.Lattice
          The second lattice.

      Returns
      -------
      (is_superlattice_of, T): (bool, numpy.ndarray[numpy.float64[3, 3]])
          Returns tuple with a boolean that is True if lattice1 is a
          superlattice of lattice2, and the tranformation matrix T
          such that L1 = L2 @ T. Note: If is_superlattice_of==True,
          numpy.rint(T).astype(int) can be used to round array elements to
          the nearest integer.
      )pbdoc");

  m.def("is_equivalent_superlattice_of", &is_equivalent_superlattice_of,
        py::arg("lattice1"), py::arg("lattice2"),
        py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Check if lattice1 is equivalent to a superlattice of lattice2

      If lattice1 is equivalent to a superlattice of lattice2, then

      .. code-block:: Python

          L1 = point_group[p].matrix() @ L2 @ T

      where p is the index of a point_group operation, T is an approximately
      integer tranformation matrix T, and L1 and L2 are the lattice vectors, as
      columns of a matrix, of lattice1 and lattice2, respectively.

      Parameters
      ----------
      lattice1 : casm.xtal.Lattice
          The first lattice.
      lattice2 : casm.xtal.Lattice
          The second lattice.
      point_group : List[casm.xtal.SymOp]
          The point group symmetry that generates equivalent lattices. Depending
          on the use case, this is often the prim crystal point group,
          :func:`~casm.xtal.make_crystal_point_group()`, or the lattice
          point group, :func:`~casm.xtal.make_point_group()`.

      Returns
      -------
      (is_equivalent_superlattice_of, T, p): (bool,
      numpy.ndarray[numpy.float64[3, 3]], int)
          Returns tuple with a boolean that is True if lattice1 is
          equivalent to a superlattice of lattice2, the
          tranformation matrix T, and point group index, p, such that L1 =
          point_group[p].matrix() @ L2 @ T. Note: If
          is_equivalent_superlattice_of==True, numpy.rint(T).astype(int) can
          be used to round array elements to the nearest integer.
      )pbdoc");

  m.def("make_transformation_matrix_to_super",
        &make_transformation_matrix_to_super, py::arg("superlattice"),
        py::arg("unit_lattice"),
        R"pbdoc(
     Returns the integer transformation matrix for the superlattice relative a unit lattice.

     Parameters
     ----------
     superlattice : casm.xtal.Lattice
         The superlattice.
     unit_lattice : casm.xtal.Lattice
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
      unit_lattice : casm.xtal.Lattice
          The unit lattice.
      point_group : List[casm.xtal.SymOp]
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
      superlattices : List[casm.xtal.Lattice]
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
      lattices : List[casm.xtal.Lattice]
          List of lattices.
      mode : str, default="commensurate"
          One of:

          - "commensurate": Returns the smallest possible superlattice of all input lattices
          - "minimal_commensurate": Returns the lattice that is the smallest possible superlattice of an equivalent lattice to all input lattice
          - "fully_commensurate": Returns the lattice that is a superlattice of all equivalents of
            all input lattices
      point_group : List[casm.xtal.symop], default=[]
          Point group that generates the equivalent lattices for the the "minimal_commensurate" and
          "fully_commensurate" modes.

      Returns
      -------
      superduperlattice : casm.xtal.Lattice
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
      properties : Dict[str, array_like]
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
      atoms : List[casm.xtal.AtomComponent], optional
          The atomic components of a molecular occupant. Atoms and
          vacancies are represented with a single AtomComponent with the
          same name for the Occupant and the AtomComponent. If atoms is
          an empty list (the default value), then an atom or vacancy is
          created, based on the name parameter.
      is_divisible : bool, default=False
          If True, indicates an Occupant that may split into components
          during kinetic Monte Carlo calculations.
      properties : Dict[str, array_like], default={}
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
           "Returns the fixed properties of the occupant");

  m.def("is_vacancy", &xtal::Molecule::is_vacancy,
        "True if occupant is a vacancy.");

  m.def("is_atomic", &xtal::Molecule::is_atomic,
        "True if occupant is a single isotropic atom or vacancy");

  m.def("make_vacancy", &xtal::Molecule::make_vacancy, R"pbdoc(
      Construct a Occupant object representing a vacancy

      This function is equivalent to ``casm.xtal.Occupant("Va")``.
      )pbdoc");

  m.def("make_atom", &xtal::Molecule::make_atom, py::arg("name"), R"pbdoc(
      Construct a Occupant object representing a single isotropic atom

      This function is equivalent to ``casm.xtal.Occupant(name)``.

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
      axis_names : List[str], default=[]
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
  // `make_prim` used for the casm.xtal.Prim __init__ method, which it appears
  // must return `std::shared_ptr<xtal::BasicStructure>`.

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
      lattice : Lattice
          The primitive cell Lattice.
      coordinate_frac : array_like, shape (3, n)
          Basis site positions, as columns of a matrix, in fractional
          coordinates with respect to the lattice vectors.
      occ_dof : List[List[str]]
          Labels of occupants allowed on each basis site. The value
          occ_dof[b] is the list of occupants allowed on the `b`-th basis
          site. The values may either be (i) the name of an isotropic atom
          (i.e. "Mg") or vacancy ("Va"), or (ii) a key in the occupants
          dictionary (i.e. "H2O", or "H2_xx"). The names are case
          sensitive, and "Va" is reserved for vacancies.
      local_dof : List[List[DoFSetBasis]], default=[[]]
          Continuous DoF allowed on each basis site. No effect if empty.
          If not empty, the value local_dof[b] is a list of :class:`DoFSetBasis`
          objects describing the DoF allowed on the `b`-th basis site.
      global_dof : List[DoFSetBasis], default=[]
          Global continuous DoF allowed for the entire crystal.
      occupants : Dict[str, Occupant], default=[]
          :class:`Occupant` allowed in the crystal. The keys are labels
          used in the occ_dof parameter. This may include isotropic
          atoms, vacancies, atoms with fixed anisotropic properties, and
          molecular occupants. A seperate key and value is required for
          all species with distinct anisotropic properties (i.e. "H2_xy",
          "H2_xz", and "H2_yz" for distinct orientations, or "A.up", and
          "A.down" for distinct collinear magnetic spins, etc.).
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
           "Returns the labels of occupants allowed on each basis site")
      .def("local_dof", &get_prim_local_dof,
           "Returns the continuous DoF allowed on each basis site")
      .def(
          "global_dof", &get_prim_global_dof,
          "Returns the continuous DoF allowed for the entire crystal structure")
      .def("occupants", &get_prim_molecules,
           "Returns the :class:`Occupant` allowed in the crystal.")
      .def_static(
          "from_json", &prim_from_json,
          "Construct a Prim from a JSON-formatted string. The `Prim reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/BasicStructure/>`_ documents the expected JSON "
          "format.",
          py::arg("prim_json_str"), py::arg("xtal_tol") = TOL)
      .def_static(
           "from_poscar", &prim_from_poscar,
           "Construct a Prim from poscar path provided as a string",
           py::arg("poscar_path")
              )
      .def("to_json", &prim_to_json,
           "Represent the Prim as a JSON-formatted string. The `Prim reference "
           "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
           "crystallography/BasicStructure/>`_ documents the expected JSON "
           "format.");

  m.def("_is_same_prim", &is_same_prim, py::arg("first"), py::arg("second"),
        R"pbdoc(
            Check if Prim are sharing the same data

            This is for testing purposes, it should be equivalent to
            `first is second` and `first == second`.

            Parameters
            ----------
            first : casm.xtal.Prim
                First Prim.

            second : casm.xtal.SharedPrim
                Second Prim.

            Returns
            ----------
            is_same : casm.xtal.Prim
                Returns true if Prim are sharing the same data

            )pbdoc");

  m.def("_share_prim", &share_prim, py::arg("init_prim"), R"pbdoc(
            Make a copy of a Prim - sharing same data

            This is for testing purposes.

            Parameters
            ----------
            init_prim : casm.xtal.Prim
                Initial prim.

            Returns
            ----------
            prim : casm.xtal.Prim
                A copy of the initial prim, sharing the same data.

            )pbdoc");

  m.def("_copy_prim", &copy_prim, py::arg("init_prim"), R"pbdoc(
            Make a copy of a Prim - not sharing same data

            This is for testing purposes.

            Parameters
            ----------
            init_prim : casm.xtal.Prim
                Initial prim.

            Returns
            ----------
            prim : casm.xtal.Prim
                A copy of the initial prim, not sharing the same data.

            )pbdoc");

  m.def("make_within", &make_within, py::arg("init_prim"), R"pbdoc(
            Returns an equivalent Prim with all basis site coordinates within the unit cell

            Parameters
            ----------
            init_prim : casm.xtal.Prim
                The initial prim.

            Returns
            ----------
            prim : Lattice
                The prim with all basis site coordinates within the unit cell.

            )pbdoc");

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
            init_prim : casm.xtal.Prim
                The initial prim.

            Returns
            ----------
            prim : Lattice
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
          init_prim : casm.xtal.Prim
              The initial prim.

          Returns
          ----------
          prim : Lattice
              The prim with canonical lattice.

        )pbdoc");

  m.def("make_canonical", &make_canonical_prim, py::arg("init_prim"),
        "Equivalent to :func:`~casm.xtal.make_canonical_prim`");

  m.def("asymmetric_unit_indices", &asymmetric_unit_indices, py::arg("prim"),
        R"pbdoc(
          Returns the indices of equivalent basis sites

          Parameters
          ----------
          prim : casm.xtal.Prim
              The prim.

          Returns
          -------
          asymmetric_unit_indices : List[List[int]]
              One list of basis site indices for each set of symmetrically equivalent basis sites.
              In other words, the elements of asymmetric_unit_indices[i] are the indices of the
              i-th set of basis sites which are symmetrically equivalent to each other.

          )pbdoc");

  m.def("make_prim_factor_group", &make_prim_factor_group, py::arg("prim"),
        R"pbdoc(
          Returns the factor group

          Parameters
          ----------
          prim : casm.xtal.Prim
              The prim.

          Returns
          -------
          factor_group : List[casm.xtal.SymOp]
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
          prim : casm.xtal.Prim
              The prim.

          Returns
          -------
          crystal_point_group : List[casm.xtal.SymOp]
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
           "Returns the time reversal value.");

  py::class_<xtal::SymInfo>(m, "SymInfo", R"pbdoc(
      Symmetry operation type, axis, invariant point, etc.

      )pbdoc")
      .def(py::init<xtal::SymOp const &, xtal::Lattice const &>(),
           py::arg("op"), py::arg("lattice"),
           R"pbdoc(

          Parameters
          ----------
          op : casm.xtal.SymOp
              The symmetry operation.
          lattice : casm.xtal.Lattice
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
      .def("to_json", &syminfo_to_json, R"pbdoc(
          Represent the symmetry operation information as a JSON-formatted string.

          The `Symmetry Operation Information JSON Object reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/symmetry/SymGroup/#symmetry-operation-json-object/>`_ documents JSON format, except conjugacy class and inverse operation are not currently included.
          )pbdoc");

  py::class_<xtal::SimpleStructure>(m, "Structure", R"pbdoc(
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

    The positions of atoms or molecules in the crystal state is defined by the lattice and atom coordinates or molecule coordinates. If included, strain and displacement properties, which are defined in reference to an ideal state, should be interpreted as the strain and displacement that takes the crystal from the ideal state to the state specified by the structure lattice and atom or molecule coordinates. The convention used by CASM is that displacements are applied first, and then the displaced coordinates and lattice vectors are strained.

    See the CASM `Degrees of Freedom (DoF) and Properties`_
    documentation for the full list of supported properties and their
    definitions.

    .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/

    )pbdoc")
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

    .. _prim-init:

    Parameters
    ----------
    lattice : Lattice
        The Lattice.
    atom_coordinate_frac : array_like, shape (3, n)
        Atom positions, as columns of a matrix, in fractional
        coordinates with respect to the lattice vectors.
    atom_type : List[str], size=n
        Atom type names.
    atom_properties : Dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with individual atoms, if present. Keys must be the name of a CASM-supported property type. Values are arrays with dimensions matching the standard dimension of the property type.
    mol_coordinate_frac : array_like, shape (3, n)
        Molecule positions, as columns of a matrix, in fractional
        coordinates with respect to the lattice vectors.
    mol_type : List[str], size=n
        Molecule type names.
    mol_properties : Dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with individual molecules, if present. Keys must be the name of a CASM-supported property type. Values are arrays with dimensions matching the standard dimension of the property type.
    global_properties : Dict[str,  numpy.ndarray[numpy.float64[m, 1]]], default={}
        Continuous properties associated with entire crystal, if present. Keys must be the name of a CASM-supported property type. Values are arrays with dimensions matching the standard dimension of the property type.
    )pbdoc")
      .def("lattice", &get_simplestructure_lattice, "Returns the lattice")
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
          "from_json", &simplestructure_from_json,
          "Construct a Structure from a JSON-formatted string. The `Structure "
          "reference "
          "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
          "crystallography/SimpleStructure/>`_ documents the expected JSON "
          "format.",
          py::arg("structure_json_str"))
      .def("to_json", &simplestructure_to_json,
           "Represent the Structure as a JSON-formatted string. The `Structure "
           "reference "
           "<https://prisms-center.github.io/CASMcode_docs/formats/casm/"
           "crystallography/SimpleStructure/>`_ documents the expected JSON "
           "format.");

  m.def("make_structure_factor_group", &make_simplestructure_factor_group,
        py::arg("structure"), R"pbdoc(
           Returns the factor group of an atomic structure

           Parameters
           ----------
           structure : casm.xtal.Structure
               The structure.

           Returns
           -------
           factor_group : List[casm.xtal.SymOp]
               The the set of symmery operations, with translation lying within the primitive unit
               cell, that leave the lattice vectors, atom coordinates, and atom types invariant.

           Notes
           -----
           Currently this method only considers atom coordinates and types. Molecular coordinates
           and types are not considered.

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
           structure : casm.xtal.Structure
               The structure.

           Returns
           -------
           crystal_point_group : List[casm.xtal.SymOp]
               The crystal point group is the group constructed from the structure factor group
               operations with translation vector set to zero.

           Notes
           -----
           Currently this method only considers atom coordinates and types. Molecular coordinates
           and types are not considered.
           )pbdoc");

  m.def("make_crystal_point_group", &make_simplestructure_crystal_point_group,
        py::arg("structure"),
        "Equivalent to :func:`~casm.xtal.make_structure_crystal_point_group`");

  m.def("make_superstructure", &make_superstructure,
        py::arg("transformation_matrix_to_super"), py::arg("structure"),
        R"pbdoc(
      Make a superstructure

      Parameters
      ----------
      transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
          The transformation matrix, T, relating the superstructure lattice vectors, S, to the unit structure lattice vectors, L, according to S = L @ T, where S and L are shape=(3,3)  matrices with lattice vectors as columns.
      structure: casm.xtal.Structure
          The unit structure used to form the superstructure.

      Returns
      -------
      superstructure: casm.xtal.Structure
          The superstructure.
      )pbdoc");

  m.def("make_equivalent_property_values", &make_equivalent_property_values,
        py::arg("point_group"), py::arg("x"), py::arg("property_type"),
        py::arg("basis") = Eigen::MatrixXd(0, 0), py::arg("tol") = TOL,
        R"pbdoc(
      Make the set of symmetry equivalent property values

      Parameters
      ----------
      point_group : List[casm.xtal.symop]
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
      equivalent_x: List[numpy.ndarray[numpy.float64[m, 1]]]
          A list of distinct property values, in the given basis,
          equivalent under the point group.
      )pbdoc");

  py::class_<StrainConverter>(m, "StrainConverter", R"pbdoc(
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
      .def(
          "metric",
          [](StrainConverter const &converter) { return converter.metric; },
          "Returns the strain metric name.")
      .def(
          "basis",
          [](StrainConverter const &converter) { return converter.basis; },
          R"pbdoc(
          Returns the basis used for strain metric vectors.

          Returns
          -------
          basis: array-like of shape (6, dim), optional
              The basis for E_vector, in terms of the standard basis.

                  E_vector_in_standard_basis = basis @ E_vector

          )pbdoc")
      .def(
          "dim",
          [](StrainConverter const &converter) {
            return converter.basis.cols();
          },
          R"pbdoc(
          Returns the strain space dimension.

          Returns
          -------
          dim: int
              The strain space dimension, equivalent to the number of columns
              of the basis matrix.
          )pbdoc")
      .def(
          "basis_pinv",
          [](StrainConverter const &converter) { return converter.basis_pinv; },
          R"pbdoc(
          Returns the strain metric basis pseudoinverse.

          Returns
          -------
          basis_pinv: numpy.ndarray[numpy.float64[dim, 6]]
              The pseudoinverse of the basis for E_vector.

                  E_vector = basis_pinv @ E_vector_in_standard_basis

          )pbdoc")
      .def_static("F_to_QU", &F_to_QU, py::arg("F"),
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
      .def_static("F_to_VQ", &F_to_VQ, py::arg("F"),
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
      .def("to_F", &strain_metric_vector_to_F, py::arg("E_vector"),
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
      .def("from_F", &strain_metric_vector_from_F, py::arg("F"),
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
      .def("to_standard_basis", &strain_metric_vector_in_standard_basis,
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
      .def("from_standard_basis", &strain_metric_vector_in_converter_basis,
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
      .def("to_E_matrix", &strain_metric_vector_to_matrix, py::arg("E_vector"),
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
      .def("from_E_matrix", &strain_metric_matrix_to_vector,
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
        &make_symmetry_adapted_strain_basis,
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
      symmetry_adapted_strain_basis: List[numpy.ndarray[numpy.float64[6, 6]]]
          The symmetry-adapted strain basis, :math:`B^{\vec{e}}`.
      )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
