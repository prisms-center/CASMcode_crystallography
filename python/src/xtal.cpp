#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

// nlohmann::json binding
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "pybind11_json/pybind11_json.hpp"

// CASM
#include "casm/casm_io/Log.hh"
#include "casm/casm_io/container/json_io.hh"
#include "casm/casm_io/json/InputParser_impl.hh"
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
#include "casm/crystallography/io/UnitCellCoordIO.hh"
#include "casm/crystallography/io/VaspIO.hh"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

/// CASM - Python binding code
namespace CASMpy {

using namespace CASM;

namespace _xtal_impl {

Eigen::MatrixXd pseudoinverse(Eigen::MatrixXd const &M) {
  return M.completeOrthogonalDecomposition().pseudoInverse();
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
    Index min_volume = 1, std::string dirs = std::string("abc"),
    std::optional<Eigen::Matrix3i> unit_cell = std::nullopt,
    bool diagonal_only = false, bool fixed_shape = false) {
  if (!unit_cell.has_value()) {
    unit_cell = Eigen::Matrix3i::Identity();
  }
  xtal::ScelEnumProps enum_props{min_volume,    max_volume + 1,
                                 dirs,          unit_cell.value(),
                                 diagonal_only, fixed_shape};
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

xtal::Lattice make_superduperlattice(std::vector<xtal::Lattice> lattices,
                                     std::string mode,
                                     std::vector<xtal::SymOp> point_group) {
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

jsonParser &to_json(DoFSetBasis const &dofsetbasis, jsonParser &json) {
  if (!json.is_object()) {
    throw std::runtime_error(
        "Error in to_json(DoFSetBasis): json must be an object");
  }
  json[dofsetbasis.dofname]["basis"] = dofsetbasis.basis.transpose();
  json[dofsetbasis.dofname]["axis_names"] = dofsetbasis.axis_names;
  return json;
}

/// \brief Parse a JSON object into a vector of DoFSetBasis
void parse(InputParser<std::vector<DoFSetBasis>> &parser,
           ParsingDictionary<AnisoValTraits> const *_aniso_val_dict = nullptr) {
  ParsingDictionary<AnisoValTraits> default_aniso_val_dict =
      make_parsing_dictionary<AnisoValTraits>();
  if (_aniso_val_dict == nullptr) _aniso_val_dict = &default_aniso_val_dict;

  jsonParser const &json = parser.self;
  if (!json.is_object()) {
    parser.error.insert("must be an object");
  }

  parser.value = notstd::make_unique<std::vector<DoFSetBasis>>();
  auto &dof = *parser.value;
  for (auto it = json.begin(); it != json.end(); ++it) {
    std::string dofname = it.name();

    if (!_aniso_val_dict->contains(dofname)) {
      try {
        AnisoValTraits const &traits = _aniso_val_dict->lookup(dofname);
      } catch (std::runtime_error const &e) {
        parser.insert_error(dofname, e.what());
        continue;
      }
    }
    AnisoValTraits const &traits = _aniso_val_dict->lookup(dofname);

    jsonParser const &dof_json = json[dofname];
    if (!dof_json.is_object()) {
      parser.insert_error(dofname, "must be an object");
      continue;
    }

    // if non-standard basis, both basis and axis_names are required
    // if standard basis, neither are allowed
    if (dof_json.contains("basis") != dof_json.contains("axis_names")) {
      std::stringstream msg;
      msg << "Error reading DoF input: if either is present, 'axis_names' and "
             "'basis' must both be present";
      parser.insert_error(dofname, msg.str());
      continue;
    }

    // standard basis
    if (!json.contains("axis_names")) {
      std::vector<std::string> axis_names = traits.standard_var_names();
      Eigen::MatrixXd basis =
          Eigen::MatrixXd::Identity(traits.dim(), traits.dim());
      dof.push_back(DoFSetBasis(dofname, axis_names, basis));
      continue;
    }

    // non-standard basis
    std::vector<std::string> axis_names;
    parser.require(axis_names, "axis_names");

    Eigen::MatrixXd row_vector_basis;
    parser.require(row_vector_basis, "basis");

    if (row_vector_basis.rows() != axis_names.size()) {
      std::stringstream msg;
      msg << "the number of basis vectors (" << row_vector_basis.rows() << ") "
          << "does not match the number of axis names (" << axis_names.size()
          << ")";
      parser.insert_error(dofname, msg.str());
    }

    if (row_vector_basis.cols() != traits.dim()) {
      std::stringstream msg;
      msg << "basis vector sizes (" << row_vector_basis.cols() << ") "
          << "do not match the standard dimension (" << traits.dim() << ")";
      parser.insert_error(dofname, msg.str());
    }

    dof.push_back(
        DoFSetBasis(dofname, axis_names, row_vector_basis.transpose()));
  }
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
    std::string title = std::string("prim"),
    std::optional<std::vector<Index>> labels = std::nullopt) {
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
  if (labels.has_value() && labels.value().size() != coordinate_frac.cols()) {
    throw std::runtime_error(
        "Error in make_prim: labels.has_value() && "
        "labels.value().size() != coordinate_frac.cols()");
  }
  if (labels.has_value()) {
    for (Index i = 0; i < labels.value().size(); ++i) {
      if (labels.value()[i] < 0) {
        std::stringstream msg;
        msg << "Error in make_prim: labels.value()[" << i
            << "] < 0 (=" << labels.value()[i] << ")";
        throw std::runtime_error(msg.str());
      }
    }
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

    if (labels.has_value()) {
      site.set_label(labels.value()[b]);
    }
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
  // print errors and warnings to sys.stdout
  py::scoped_ostream_redirect redirect;
  jsonParser json = jsonParser::parse(prim_json_str);
  PyErr_WarnEx(PyExc_DeprecationWarning,
               "Prim.from_json() is deprecated, use Prim.from_dict() instead.",
               2);
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

xtal::SimpleStructure _simplestructure_from_poscar(std::istream &poscar_stream,
                                                   std::string mode) {
  xtal::SimpleStructure simple =
      xtal::make_simple_structure(poscar_stream, TOL);
  if (mode == "atoms") {
    simple.mol_info.resize(0);
    simple.mol_info.properties.clear();
  } else if (mode == "molecules") {
    simple.atom_info.resize(0);
    simple.atom_info.properties.clear();
  } else if (mode != "both") {
    std::stringstream ss;
    ss << "Invalid mode: '" << mode
       << "' Must be one of 'atoms', 'molecules', or 'both'";
    throw std::runtime_error(ss.str());
  }
  return simple;
}

xtal::SimpleStructure simplestructure_from_poscar(std::string &poscar_path,
                                                  std::string mode) {
  std::filesystem::path path(poscar_path);
  std::ifstream poscar_stream(path);
  return _simplestructure_from_poscar(poscar_stream, mode);
}

xtal::SimpleStructure simplestructure_from_poscar_str(std::string &poscar_str,
                                                      std::string mode) {
  std::istringstream poscar_stream(poscar_str);
  return _simplestructure_from_poscar(poscar_stream, mode);
}

/// \brief Format xtal::BasicStructure as JSON string
std::string prim_to_json(
    std::shared_ptr<xtal::BasicStructure const> const &prim, bool frac,
    bool include_va) {
  PyErr_WarnEx(PyExc_DeprecationWarning,
               "Prim.to_json() is deprecated, use Prim.to_dict() instead.", 2);
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

std::vector<Index> get_prim_labels(
    std::shared_ptr<xtal::BasicStructure const> const &prim) {
  std::vector<Index> labels;
  for (auto const &site : prim->basis()) {
    labels.push_back(site.label());
  }
  return labels;
}

std::shared_ptr<xtal::BasicStructure const> make_within(
    std::shared_ptr<xtal::BasicStructure const> const &init_prim) {
  auto prim = std::make_shared<xtal::BasicStructure>(*init_prim);
  prim->within();
  return prim;
}

std::shared_ptr<xtal::BasicStructure const> make_primitive_prim(
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
  PyErr_WarnEx(
      PyExc_DeprecationWarning,
      "SymInfo.to_json() is deprecated, use SymInfo.to_dict() instead.", 2);

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
  // print errors and warnings to sys.stdout
  py::scoped_ostream_redirect redirect;
  PyErr_WarnEx(
      PyExc_DeprecationWarning,
      "Structure.from_json() is deprecated, use Structure.from_dict() instead.",
      2);
  jsonParser json = jsonParser::parse(json_str);
  xtal::SimpleStructure simple;
  from_json(simple, json);
  return simple;
}

std::string simplestructure_to_json(xtal::SimpleStructure const &simple) {
  PyErr_WarnEx(
      PyExc_DeprecationWarning,
      "Structure.to_json() is deprecated, use Structure.to_dict() instead.", 2);
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

xtal::SimpleStructure make_primitive_simplestructure(
    xtal::SimpleStructure const &init_structure) {
  std::vector<std::vector<std::string>> occ_dof;
  for (std::string name : init_structure.atom_info.names) {
    occ_dof.push_back({name});
  }
  std::shared_ptr<xtal::BasicStructure const> prim = make_prim(
      get_simplestructure_lattice(init_structure, TOL),
      get_simplestructure_atom_coordinate_frac(init_structure), occ_dof);
  prim = make_primitive_prim(prim);
  xtal::SimpleStructure structure;

  std::vector<std::string> atom_type;
  for (auto const &site_names : prim->unique_names()) {
    atom_type.push_back(site_names[0]);
  }

  return make_simplestructure(prim->lattice(), get_prim_coordinate_cart(prim),
                              atom_type);
}

xtal::SimpleStructure make_canonical_simplestructure(
    xtal::SimpleStructure const &init_structure) {
  xtal::SimpleStructure structure = init_structure;
  xtal::Lattice lattice = get_simplestructure_lattice(init_structure, TOL);
  lattice.make_right_handed();
  lattice = xtal::canonical::equivalent(lattice);
  structure.lat_column_mat = lattice.lat_column_mat();
  return structure;
}

xtal::SimpleStructure make_superstructure(
    py::EigenDRef<Eigen::Matrix3l const> transformation_matrix_to_super,
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
  // B * x_after = M * B * x_before
  // x_after = B_pinv * M * B * x_before
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

xtal::UnitCellCoordIndexConverter make_SiteIndexConverter(
    py::EigenDRef<Eigen::Matrix3l const> transformation_matrix_to_super,
    int n_sublattice) {
  return xtal::UnitCellCoordIndexConverter(transformation_matrix_to_super,
                                           n_sublattice);
}

xtal::UnitCellIndexConverter make_UnitCellIndexConverter(
    py::EigenDRef<Eigen::Matrix3l const> transformation_matrix_to_super) {
  return xtal::UnitCellIndexConverter(transformation_matrix_to_super);
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
        - Methods for enumerating lattices, making superstructures,
          finding primitive and reduced cells, and finding symmetry
          operations.

    )pbdoc";

  py::class_<xtal::SymOp> pySymOp(m, "SymOp", R"pbdoc(
      The Cartesian representation, :math:`\left\{ \mathbf{R}, \vec{\tau} \right\}`, of a
      symmetry operation acts on a Cartesian coordinate according to

      .. math::

          \vec{r}^{\ after} = \mathbf{R} \vec{r}^{\ before} + \vec{\tau},

      where :math:`\mathbf{R}` is a 3x3 matrix, :math:`\vec{\tau}` is a Cartesian
      translation vector, :math:`\vec{r}^{\ before}` is the Cartesian coordinate before
      application of symmetry, and :math:`\vec{r}^{\ after}` is the Cartesian
      coordinate after application of symmetry.

      In libcasm-xtal this is represented by a SymOp, `op`, that transforms a
      Cartesian coordinate according to

      .. code-block:: Python

          r_after = op.matrix() @ r_before + op.translation()

      where `r_before` and `r_after` are shape=(3,) arrays with the Cartesian
      coordinates before and after transformation, respectively.

      Additionally, a symmetry operation may include a flip in the sign of magnetic
      spins. In `libcasm-xtal`, the sign of magnetic spins is flipped according to:

      .. code-block:: Python

          if op.time_reversal() is True:
              s_after = -s_before

      where `s_before` and `s_after` are the spins before and after
      transformation, respectively.

      .. rubric:: Special Methods

      The multiplication operator ``X = lhs * rhs`` can be used to apply SymOp to
      various objects:

      - ``X=SymOp``, ``lhs=SymOp``, ``rhs=SymOp``: Construct the
        :class:`SymOp`, `X`, equivalent to applying first `rhs`, then
        `lhs`.
      - ``X=np.ndarray``, ``lhs=SymOp``, ``rhs=np.ndarray``: Transform multiple
        Cartesian coordinates, represented as columns of a `np.ndarray`.
      - ``X=dict[str,np.ndarray]``, ``lhs=SymOp``, ``rhs=dict[str,np.ndarray]``:
        Transform CASM-supported properties (local or global). Keys must be the name
        of a CASM-supported property type. Values are arrays with the number of rows
        matching the standard dimension of the property type. For local properties,
        columns correspond to the value associated with each site. For global
        properties, the input arrays may be of shape `(m,)` or `(m,1)`, where `m` is
        the CASM standard dimension, and the output arrays will always be shape
        `(m,1)`. See the CASM `Degrees of Freedom (DoF) and Properties`_ documentation
        for the full list of supported properties and their definitions.
      - ``X=Lattice``, ``lhs=SymOp``, ``rhs=Lattice``: Transform a
        :class:`Lattice`.
      - ``X=Structure``, ``lhs=SymOp``, ``rhs=Structure``: Transform a
        :class:`Structure`.

      - SymOp may be copied with :func:`SymOp.copy <libcasm.xtal.SymOp.copy>`,
        `copy.copy`, or `copy.deepcopy`.

      .. note::

          Other types of objects require additional information to be efficiently
          transformed by a symmetry operation. Some other symmetry representations used
          in CASM include:

          - :class:`IntegralSiteCoordinateRep`: Transform
            :class:`IntegralSiteCoordinate` and
            :class:`~libcasm.clusterography.Cluster`.
          - :class:`~libcasm.configuration.SupercellSymOp`: Transform
            :class:`~libcasm.configuration.Configuration`
          - :class:`~libcasm.occ_events.OccEventRep`: Transform
            :class:`~libcasm.occ_events.OccEvent`
          - Matrix representations (``numpy.ndarray``): Transform vectors of degree of
            freedom (DoF) values, order parameters, etc.
          - Permutation arrays (``list[int]``): Transform site indices


      .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/

      )pbdoc");

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

    Atom representation is most widely supported in CASM methods. In some limited cases
    the molecule representation is used.

    Notes
    -----

    The positions of atoms or molecules in the crystal structure is defined by the
    lattice and atom coordinates or molecule coordinates. If included, strain and
    displacement properties, which are defined in reference to an ideal state, should
    be interpreted as the strain and displacement that takes the crystal from the ideal
    state to the state specified by the structure lattice and atom or molecule 
    coordinates. The convention used by CASM is that displacements are applied first,
    and then the displaced coordinates and lattice vectors are strained.

    See the CASM `Degrees of Freedom (DoF) and Properties`_
    documentation for the full list of supported properties and their
    definitions.

    .. rubric:: Special Methods

    - Structure may be copied with :func:`Structure.copy <libcasm.xtal.Structure.copy>`,
      `copy.copy`, or `copy.deepcopy`.


    .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/

    )pbdoc");

  py::class_<xtal::Lattice>(m, "Lattice", R"pbdoc(
      A 3-dimensional lattice

      .. rubric:: Special Methods

      Sort :class:`Lattice` by how canonical the lattice vectors are
      via ``<``, ``<=``, ``>``, ``>=`` (see also
      :ref:`Lattice Canonical Form <lattice-canonical-form>`), and check if lattice
      are approximately equal via ``==``, ``!=``:

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

      - Lattice may be copied with
        :func:`Lattice.copy <libcasm.xtal.Lattice.copy>`,
        `copy.copy`, or `copy.deepcopy`.


      .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/
      )pbdoc")
      .def(py::init<Eigen::Matrix3d const &, double>(), "Construct a Lattice",
           py::arg("column_vector_matrix"), py::arg("tol") = TOL, R"pbdoc(

      .. rubric:: Constructor

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
      .def("reciprocal", &xtal::Lattice::reciprocal,
           "Return the reciprocal lattice")
      .def(
          "lengths_and_angles",
          [](xtal::Lattice const &self) -> std::vector<double> {
            std::vector<double> v;
            v.push_back(self.length(0));
            v.push_back(self.length(1));
            v.push_back(self.length(2));
            v.push_back(self.angle(0));
            v.push_back(self.angle(1));
            v.push_back(self.angle(2));
            return v;
          },
          R"pbdoc(
           Return the lattice vector lengths and angles,
           :math:`[a, b, c, \alpha, \beta, \gamma]`.
           )pbdoc")
      .def("volume", &xtal::Lattice::volume, R"pbdoc(
           Return the signed volume of the unit cell.
           )pbdoc")
      .def("voronoi_table", &xtal::Lattice::voronoi_table, R"pbdoc(
           Return the Voronoi table for the lattice.

           Returns
           -------
           voronoi_table : numpy.ndarray[numpy.float64[n, 3]]
               The Voronoi table, where each row is an outward-pointing normal
               of the lattice Voronoi cell, defined such that if
               ``np.max(voronoi_table @ cart_coord) > 1.0``, then
               `cart_coord` is outside of the Voronoi cell.
           )pbdoc")
      .def("voronoi_inner_radius", &xtal::Lattice::inner_voronoi_radius,
           R"pbdoc(
           Return the radius of the largest sphere that fits wholly within the
           Voronoi cell
           )pbdoc")
      .def(
          "voronoi_number",
          [](xtal::Lattice const &lattice, Eigen::Vector3d const &pos) {
            int tnum = 0;
            double tproj = 0;

            Eigen::MatrixXd const &vtable = lattice.voronoi_table();

            for (Index nv = 0; nv < vtable.rows(); nv++) {
              tproj = vtable.row(nv) * pos;
              if (almost_equal(tproj, 1.0, lattice.tol())) {
                tnum++;
              } else if (tproj > 1.0) {
                return -1;
              }
            }

            return tnum;
          },
          R"pbdoc(
          Return the number of lattice points that `pos` is equally as
          close to as the origin

          Parameters
          ----------
          pos : array_like, shape=(3,)
              The position to check, in Cartesian coordinates.

          Returns
          -------
          n : int
              If `n` is 0, then `pos` is within the Voronoi cell and the origin
              is the nearest lattice site. If `n` is -1, then `pos` is outside
              the Voronoi cell and there is a lattice site closer than the
              origin. If `n` is in [1, 7], then `pos` is equally as close to
              `n` lattice sites as the origin.
          )pbdoc",
          py::arg("pos"))
      .def_static("from_lengths_and_angles",
                  &xtal::Lattice::from_lengths_and_angles,
                  py::arg("lengths_and_angles"), py::arg("tol") = TOL,
                  R"pbdoc(
            Construct a Lattice from the lattice vector lengths and angles,
            :math:`[a, b, c, \alpha, \beta, \gamma]`
            )pbdoc")
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
      .def("is_equivalent_to", &lattice_is_equivalent_to, py::arg("lattice2"),
           R"pbdoc(
            Check if this lattice is equivalent to another lattice

            Two lattices, L1 and L2, are equivalent in the sense that the
            lattice points have the same Cartesian coordinates if there exists
            `U` such that:

            .. code-block:: Python

                L1 = L2 @ U,

            where `L1` and `L2` are the Cartesian lattice vectors as matrix
            columns, and `U` is a unimodular matrix (integer matrix, with
            abs(det(U))==1).

            Notes
            -----

            - Use :func:`libcasm.mapping.methods.map_lattices` to check if
              lattices are equivalent in the sense that a rigid rotation can
              map the Cartesian coordinates of one lattice onto the other.

            Parameters
            ----------
            lattice2 : Lattice
                The second lattice.

            Returns
            -------
            is_equivalent: bool
                True if `self`, with lattice vectors `L1`, is equivalent to
                `lattice2`, with lattice vectors `L2`, in the sense that
                the lattice points have the same Cartesian coordinates; False
                otherwise.
            )pbdoc")
      .def("is_superlattice_of", &is_superlattice_of, py::arg("lattice2"),
           R"pbdoc(
      Check if this lattice is a superlattice of another lattice

      If `lattice1` (self) is a superlattice of `lattice2`, then

      .. code-block:: Python

          L1 = L2 @ T

      where `T` is an approximately integer tranformation matrix, and `L1` and `L2`
      are the lattice vectors of `lattice1` and `lattice2`, respectively, as column
      vector matrices.

      Parameters
      ----------
      lattice2 : Lattice
          The second lattice.

      Returns
      -------
      is_superlattice_of: bool
          True if `lattice1` (`self`) is a superlattice of `lattice2`; False otherwise.

      T: numpy.ndarray[numpy.float64[3, 3]]
          The value of the tranformation matrix, `T`, is such that ``L1 = L2 @ T``.
          Note that `T` is returned as an array of float, but if
          ``is_superlattice_of==True``, then ``numpy.rint(T).astype(int)``
          can be used to round array elements to the nearest integer and construct an
          array of int.

      )pbdoc")
      .def("is_equivalent_superlattice_of", &is_equivalent_superlattice_of,
           py::arg("lattice2"),
           py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Check if this lattice is equivalent to a superlattice of another lattice

      If `lattice1` (`self`) is equivalent to a superlattice of `lattice2`, then

      .. code-block:: Python

          L1 = point_group[p].matrix() @ L2 @ T

      where `p` is the index of a `point_group` operation, `T` is an approximately
      integer tranformation matrix, and `L1` and `L2` are the lattice vectors of
      `lattice1` and `lattice2`, respectively, as column vector matrices.

      Parameters
      ----------
      lattice2 : Lattice
          The second lattice.
      point_group : list[SymOp]
          The point group symmetry that generates equivalent lattices. Depending
          on the use case, this is often the prim crystal point group,
          :func:`make_crystal_point_group()`, or the lattice point group,
          :func:`make_point_group()`.

      Returns
      -------
      is_equivalent_superlattice_of: bool
          True if `lattice1` (`self`) is equivalent to a superlattice of `lattice2`;
          False otherwise.

      T: numpy.ndarray[numpy.float64[3, 3]]
          If ``is_equivalent_superlattice_of==True``, then the values of the
          tranformation matrix, `T`, and point group index, `p`, are such that
          ``L1 = point_group[p].matrix() @ L2 @ T``; otherwise `T` has undefined
          value. Note that `T` is returned as an array of float, but if
          ``is_equivalent_superlattice_of==True``, then ``numpy.rint(T).astype(int)``
          can be used to round array elements to the nearest integer and construct an
          array of int.

      p: int
          If ``is_equivalent_superlattice_of==True``, `p` is the index of the first
          :class:`SymOp` into `point_group` for which it is true; otherwise the value
          of `p` is undefined.
      )pbdoc")
      .def(
          "copy", [](xtal::Lattice const &self) { return xtal::Lattice(self); },
          R"pbdoc(
          Returns a copy of the Lattice.
          )pbdoc")
      .def("__copy__",
           [](xtal::Lattice const &self) { return xtal::Lattice(self); })
      .def("__deepcopy__", [](xtal::Lattice const &self,
                              py::dict) { return xtal::Lattice(self); })
      .def("__repr__",
           [](xtal::Lattice const &self) {
             std::stringstream ss;
             jsonParser json;
             json["lattice_vectors"] = self.lat_column_mat().transpose();
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",  // Lattice.from_dict
          [](const nlohmann::json &data, double xtal_tol) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

            jsonParser json{data};
            Eigen::Matrix3d latvec_transpose;
            try {
              from_json(latvec_transpose, json["lattice_vectors"]);
            } catch (std::exception &e) {
              log() << e.what() << std::endl;
              throw std::runtime_error("Error parsing Lattice from JSON.");
            }
            return xtal::Lattice(latvec_transpose.transpose(), xtal_tol);
          },
          R"pbdoc(
          Construct a Lattice from a Python dict

          Expected format:

          .. code-block:: Python

              data = {
                  "lattice_vectors": [
                      [a1, a2, a3],
                      [b1, b2, b3],
                      [c1, c2, c3],
                  ]
              }

          Parameters
          ----------
          data: dict
              The Python dict representation of the Lattice.
          xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
              The tolerance used for crystallographic comparisons.

          Returns
          -------
          lattice: Lattice
              The Lattice.

          )pbdoc",
          py::arg("data"), py::arg("xtal_tol") = TOL)
      .def(
          "to_dict",
          [](xtal::Lattice const &lattice) {
            jsonParser json;
            json["lattice_vectors"] = lattice.lat_column_mat().transpose();
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the Lattice as a Python dict

          The Lattice dict format:

          .. code-block:: Python

              data = {
                  "lattice_vectors": [
                      [a1, a2, a3],
                      [b1, b2, b3],
                      [c1, c2, c3],
                  ]
              }

          Returns
          -------
          data: dict
              The Lattice as a Python dict.

          )pbdoc");

  m.def("make_canonical_lattice", &make_canonical_lattice,
        py::arg("init_lattice"),
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
    init_lattice : Lattice
        The initial lattice.

    Returns
    -------
    lattice : Lattice
        The canonical equivalent lattice, using the lattice point group.
    )pbdoc");

  m.def("make_canonical", &make_canonical_lattice, py::arg("init_lattice"),
        "Equivalent to :func:`make_canonical_lattice`");

  m.def("fractional_to_cartesian", &fractional_to_cartesian, py::arg("lattice"),
        py::arg("coordinate_frac"), R"pbdoc(
    Convert fractional coordinates to Cartesian coordinates

    The result is equal to:

    .. code-block:: Python

        lattice.column_vector_matrix() @ coordinate_frac

    Parameters
    ----------
    lattice : Lattice
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
    lattice : Lattice
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
    lattice : Lattice
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
      Return minimum length displacement (:math:`r_2` - :math:`r_1`), accounting for
      periodic boundary conditions.

      Parameters
      ----------
      lattice : Lattice
          The lattice, defining the periodic boundaries.
      r1 : array_like, shape (3, 1)
          Position, :math:`r_1`, in Cartesian coordinates.
      r2 : array_like, shape (3, 1)
          Position, :math:`r_2`, in Cartesian coordinates.
      robust : boolean, default=True
          If True, use a "robust" method which uses the lattice's Wigner-Seitz
          cell to determine the nearest image, which guarantees to find the
          minimum distance. If False, use a "fast" method, which removes integer
          multiples of lattice translations from the displacement, but may not
          result in the true minimum distance.

      Returns
      -------
      displacement: numpy.ndarray[numpy.float64[3, 1]]]
          The displacement, :math:`r_2` - :math:`r_1`, in Cartesian coordinates, with
          minimum length, accounting for periodic boundary conditions.
      )pbdoc");

  m.def("make_point_group", &make_lattice_point_group, py::arg("lattice"),
        R"pbdoc(
      Returns the lattice point group

      Parameters
      ----------
      lattice : Lattice
          The lattice.

      Returns
      -------
      point_group : list[SymOp]
          The set of rigid transformations that keep the origin fixed
          (i.e. have zero translation vector) and map the lattice (i.e.
          all points that are integer multiples of the lattice vectors)
          onto itself.
      )pbdoc");

  m.def("make_transformation_matrix_to_super",
        &make_transformation_matrix_to_super, py::arg("superlattice"),
        py::arg("unit_lattice"),
        R"pbdoc(
     Returns the integer transformation matrix for the superlattice relative a unit
     lattice.

     Parameters
     ----------
     superlattice : Lattice
         The superlattice.
     unit_lattice : Lattice
         The unit lattice.

     Returns
     -------
     T: numpy.ndarray[numpy.int64[3, 3]]
         Returns the integer tranformation matrix `T` such that ``S = L @ T``, where
         `S` and `L` are the lattice vectors, as columns of a matrix, of `superlattice`
         and `unit_lattice`, respectively.

     Raises
     ------
     RuntimeError:
         If `superlattice` is not a superlattice of `unit_lattice`.
     )pbdoc");

  m.def("enumerate_superlattices", &enumerate_superlattices,
        py::arg("unit_lattice"), py::arg("point_group"), py::arg("max_volume"),
        py::arg("min_volume") = Index(1), py::arg("dirs") = std::string("abc"),
        py::arg("unit_cell") = std::nullopt, py::arg("diagonal_only") = false,
        py::arg("fixed_shape") = false,
        R"pbdoc(
      Enumerate symmetrically distinct superlattices

      Superlattices satify:

      .. code-block:: Python

          S = L @ T,

      where `S` and `L` are, respectively, the superlattice and unit lattice vectors as
      columns of shape=(3, 3) matrices, and `T` is an integer shape=(3,3)
      transformation matrix.

      Superlattices `S1` and `S2` are symmetrically equivalent if there exists `p` and
      `A` such that:

      .. code-block:: Python

          S2 = p.matrix() @ S1 @ A,

      where `p` is an element in the point group, and `A` is a unimodular matrix
      (integer matrix, with abs(det(A))==1).

      Parameters
      ----------
      unit_lattice : Lattice
          The unit lattice.
      point_group : list[SymOp]
          The point group symmetry that determines if superlattices are equivalent.
          Depending on the use case, this is often the prim crystal point group,
          :func:`make_crystal_point_group()`, or the lattice point group,
          :func:`make_point_group()`.
      max_volume : int
          The maximum volume superlattice to enumerate. The volume is measured
          relative the unit cell being used to generate supercells.
      min_volume : int, default=1
          The minimum volume superlattice to enumerate. The volume is measured
          relative the unit cell being used to generate supercells.
      dirs : str, default="abc"
          A string indicating which lattice vectors to enumerate over. Some combination
          of 'a', 'b', and 'c', where 'a' indicates the first lattice vector of the
          unit cell, 'b' the second, and 'c' the third.
      unit_cell: Optional[np.ndarray] = None,
          An integer shape=(3,3) transformation matrix `U` allows
          specifying an alternative unit cell that can be used to generate
          superlattices of the form `S = (L @ U) @ T`. If None, `U` is set to the
          identity matrix.
      diagonal_only: bool = False
          If true, restrict :math:`T` to diagonal matrices.
      fixed_shape: bool = False
          If true, restrict `T` to diagonal matrices with diagonal coefficients
          `[m, 1, 1]` (1d), `[m, m, 1]` (2d), or `[m, m, m]` (3d),
          where the dimension is determined from ``len(dirs)``.

      Returns
      -------
      superlattices : list[Lattice]
          A list of superlattices of the `unit_lattice` which are distinct under
          application of `point_group`. The resulting lattices will be in canonical
          form with respect to `point_group`.
      )pbdoc");

  m.def("make_superduperlattice", &make_superduperlattice, py::arg("lattices"),
        py::arg("mode") = std::string("commensurate"),
        py::arg("point_group") = std::vector<xtal::SymOp>{}, R"pbdoc(
      Returns the smallest lattice that is superlattice of the input lattices

      Parameters
      ----------

      lattices : list[:class:`Lattice`]
          List of lattices.

      mode : str, default="commensurate"
          One of:

          - "commensurate": Returns the smallest possible superlattice of all input lattices
          - "minimal_commensurate": Returns the lattice that is the smallest possible superlattice of an equivalent lattice to all input lattice
          - "fully_commensurate": Returns the lattice that is a superlattice of all equivalents of
            all input lattices

      point_group : list[:class:`SymOp`], default=[]
          Point group that generates the equivalent lattices for the the "minimal_commensurate" and
          "fully_commensurate" modes.

      Returns
      -------
      superduperlattice : Lattice
          The superduperlattice
      )pbdoc");

  py::class_<xtal::AtomPosition>(m, "AtomComponent", R"pbdoc(
      An atomic component of a molecular :class:`Occupant`

      .. rubric:: Special Methods

      - AtomComponent may be copied with
        :func:`AtomComponent.copy <libcasm.xtal.AtomComponent.copy>`,
        `copy.copy`, or `copy.deepcopy`.


      )pbdoc")
      .def(py::init(&make_atom_position), py::arg("name"),
           py::arg("coordinate"), py::arg("properties"), R"pbdoc(

      .. rubric:: Constructor

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

           The position is in Cartesian coordinates, relative to the
           basis site at which the occupant containing this atom
           is placed.
           )pbdoc")
      .def("properties", &get_atom_position_properties,
           "Returns the fixed properties of the atom")
      .def(
          "copy",
          [](xtal::AtomPosition const &self) {
            return xtal::AtomPosition(self);
          },
          R"pbdoc(
          Returns a copy of the AtomComponent.
          )pbdoc")
      .def("__copy__",
           [](xtal::AtomPosition const &self) {
             return xtal::AtomPosition(self);
           })
      .def("__deepcopy__", [](xtal::AtomPosition const &self,
                              py::dict) { return xtal::AtomPosition(self); })
      .def("__repr__",
           [](xtal::AtomPosition const &self) {
             // Write in Cartesian coordinates
             std::stringstream ss;
             jsonParser json;
             to_json(self, json, Eigen::Matrix3d::Identity());
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",  // AtomComponent.from_dict
          [](const nlohmann::json &data, bool frac,
             std::optional<xtal::Lattice> lattice) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

            jsonParser json{data};

            Eigen::Matrix3d coord_mode_to_cart_M;
            if (frac) {
              if (!lattice.has_value()) {
                throw std::runtime_error(
                    "Error in AtomComponent.from_dict: Reading fractional "
                    "coordinates requires a lattice.");
              }
              coord_mode_to_cart_M = lattice->lat_column_mat();
            } else {
              coord_mode_to_cart_M = Eigen::Matrix3d::Identity();
            }

            ParsingDictionary<AnisoValTraits> default_aniso_val_dict =
                make_parsing_dictionary<AnisoValTraits>();
            return jsonConstructor<xtal::AtomPosition>::from_json(
                json, coord_mode_to_cart_M, default_aniso_val_dict);
          },
          R"pbdoc(
          Construct an AtomComponent from a Python dict

          The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/#atom-component-json-object>`_
          documents the expected format:

          .. code-block:: Python

              data = {
                  "coordinate": [r1, r2, r3],
                  "name": "<atom_name>",
                  "properties": {
                    "<property_name>": {
                      "value": [v1, ...],
                    },
                    ...
                  }
              }

          Parameters
          ----------
          data: dict
              The Python dict representation of the AtomComponent
          frac: bool = False
              If True, read coordinate as fractional.
          lattice: Optional[Lattice] = None
              The lattice, required if `frac` is True.

          Returns
          -------
          lattice: Lattice
              The Lattice.

          )pbdoc",
          py::arg("data"), py::arg("frac") = false,
          py::arg("lattice") = std::nullopt)
      .def(
          "to_dict",
          [](xtal::AtomPosition const &self, bool frac,
             std::optional<xtal::Lattice> lattice) {
            Eigen::Matrix3d cart_to_coord_mode_M;
            if (frac) {
              if (!lattice.has_value()) {
                throw std::runtime_error(
                    "Error in AtomComponent.to_dict: Writing fractional "
                    "coordinates requires a lattice.");
              }
              cart_to_coord_mode_M = lattice->inv_lat_column_mat();
            } else {
              cart_to_coord_mode_M = Eigen::Matrix3d::Identity();
            }
            jsonParser json;
            to_json(self, json, cart_to_coord_mode_M);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the AtomComponent as a Python dict

          The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/#atom-component-json-object>`_
          documents the AtomComponent format.

          Parameters
          ----------
          frac: bool = False
              If True, write fractional coordinates.
          lattice: Optional[Lattice] = None
              The lattice, required if `frac` is True.

          Returns
          -------
          data: dict
              The AtomComponent as a Python dict.

          )pbdoc",
          py::arg("frac") = false, py::arg("lattice") = std::nullopt);

  py::class_<xtal::Molecule>(m, "Occupant", R"pbdoc(
      A site occupant, which may be a vacancy, atom, or molecule

      The Occupant class is used to represent all chemical species,
      including single atoms, vacancies, and molecules.

      .. rubric:: Special Methods

      - Occupant may be copied with
        :func:`Occupant.copy <libcasm.xtal.Occupant.copy>`,
        `copy.copy`, or `copy.deepcopy`.

      )pbdoc")
      .def(py::init(&make_molecule), py::arg("name"),
           py::arg("atoms") = std::vector<xtal::AtomPosition>{},
           py::arg("is_divisible") = false,
           py::arg("properties") = std::map<std::string, Eigen::MatrixXd>{},
           R"pbdoc(

      .. rubric:: Constructor

      Parameters
      ----------
      name : str
          A \"chemical name\", which must be identical for occupants to
          be found symmetrically equivalent. The names are case
          sensitive, and "Va" is reserved for vacancies.
      atoms : list[:class:`AtomComponent`], optional
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
           "True if occupant is divisible in kinetic Monte Carlo calculations")
      .def("atoms", &xtal::Molecule::atoms,
           "Returns the atomic components of the occupant")
      .def("properties", &get_molecule_properties,
           "Returns the fixed properties of the occupant")
      .def("is_vacancy", &xtal::Molecule::is_vacancy,
           "True if occupant is a vacancy.")
      .def("is_atomic", &xtal::Molecule::is_atomic,
           "True if occupant is a single isotropic atom or vacancy")
      .def(
          "copy",
          [](xtal::Molecule const &self) { return xtal::Molecule(self); },
          R"pbdoc(
          Returns a copy of the Lattice.
          )pbdoc")
      .def("__copy__",
           [](xtal::Molecule const &self) { return xtal::Molecule(self); })
      .def("__deepcopy__", [](xtal::Molecule const &self,
                              py::dict) { return xtal::Molecule(self); })
      .def("__repr__",
           [](xtal::Molecule const &self) {
             // Write in Cartesian coordinates
             std::stringstream ss;
             jsonParser json;
             to_json(self, json, Eigen::Matrix3d::Identity());
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",  // Occupant.from_dict
          [](const nlohmann::json &data, bool frac,
             std::optional<xtal::Lattice> lattice) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

            jsonParser json{data};

            Eigen::Matrix3d coord_mode_to_cart_M;
            if (frac) {
              if (!lattice.has_value()) {
                throw std::runtime_error(
                    "Error in Occupant.from_dict: Reading fractional "
                    "coordinates requires a lattice.");
              }
              coord_mode_to_cart_M = lattice->lat_column_mat();
            } else {
              coord_mode_to_cart_M = Eigen::Matrix3d::Identity();
            }

            ParsingDictionary<AnisoValTraits> default_aniso_val_dict =
                make_parsing_dictionary<AnisoValTraits>();
            return jsonConstructor<xtal::Molecule>::from_json(
                json, coord_mode_to_cart_M, default_aniso_val_dict);
          },
          R"pbdoc(
          Construct an Occupant from a Python dict

          The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/#molecule-json-object>`_
          documents the Occupant / Molecule format:

          .. code-block:: Python

              data = {
                  "atoms": [
                    {... AtomComponent ... },
                    ...
                  ],
                  "name": "<chemical_name>",
                  "properties": {
                    "<property_name>": {
                      "value": [v1, ...],
                    },
                    ...
                  }
              }

          Parameters
          ----------
          data: dict
              The Python dict representation of the AtomComponent
          frac: bool = False
              If True, read coordinate as fractional.
          lattice: Optional[Lattice] = None
              The lattice, required if `frac` is True.

          Returns
          -------
          lattice: Lattice
              The Lattice.

          )pbdoc",
          py::arg("data"), py::arg("frac") = false,
          py::arg("lattice") = std::nullopt)
      .def(
          "to_dict",
          [](xtal::Molecule const &self, bool frac,
             std::optional<xtal::Lattice> lattice) {
            Eigen::Matrix3d cart_to_coord_mode_M;
            if (frac) {
              if (!lattice.has_value()) {
                throw std::runtime_error(
                    "Error in Occupant.to_dict: Writing fractional "
                    "coordinates requires a lattice.");
              }
              cart_to_coord_mode_M = lattice->inv_lat_column_mat();
            } else {
              cart_to_coord_mode_M = Eigen::Matrix3d::Identity();
            }
            jsonParser json;
            to_json(self, json, cart_to_coord_mode_M);
            return static_cast<nlohmann::json>(json);
          },
          R"pbdoc(
          Represent the Occupant as a Python dict

          The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/#molecule-json-object>`_
          documents the Occupant / Molecule format.

          Parameters
          ----------
          frac: bool = False
              If True, write fractional coordinates.
          lattice: Optional[Lattice] = None
              The lattice, required if `frac` is True.

          Returns
          -------
          data: dict
              The Occupant as a Python dict.

          )pbdoc",
          py::arg("frac") = false, py::arg("lattice") = std::nullopt);

  m.def("make_vacancy", &xtal::Molecule::make_vacancy, R"pbdoc(
      Construct a Occupant object representing a vacancy

      This function is equivalent to ``Occupant("Va")``.
      )pbdoc");

  m.def("make_atom", &xtal::Molecule::make_atom, py::arg("name"), R"pbdoc(
      Construct a Occupant object representing a single isotropic atom

      This function is equivalent to ``Occupant(name)``.

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

      .. rubric:: Special Methods

      - DoFSetBasis may be copied with
        :func:`DoFSetBasis.copy <libcasm.xtal.Occupant.copy>`,
        `copy.copy`, or `copy.deepcopy`.

      .. _`Degrees of Freedom (DoF) and Properties`: https://prisms-center.github.io/CASMcode_docs/formats/dof_and_properties/
      )pbdoc")
      .def(py::init(&make_dofsetbasis), py::arg("dofname"),
           py::arg("axis_names") = std::vector<std::string>{},
           py::arg("basis") = Eigen::MatrixXd(0, 0), R"pbdoc(

      .. rubric:: Constructor

      Parameters
      ----------
      dofname : str
          The type of DoF. Must be a CASM supported DoF type.
      basis : array_like, shape (m, n), default=numpy.ndarray[numpy.float64[1, 0]]
          User-specified DoF basis vectors, as columns of a matrix. The
          DoF values in this basis, `x_prim`, are related to the DoF
          values in the CASM standard basis, `x_standard`, according to
          ``x_standard = basis @ x_prim``. The number of rows in the basis
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
      .def("basis", &get_dofsetbasis_basis, "Returns the basis matrix.")
      .def(
          "copy", [](DoFSetBasis const &self) { return DoFSetBasis(self); },
          R"pbdoc(
          Returns a copy of the DoFSetBasis.
          )pbdoc")
      .def("__copy__",
           [](DoFSetBasis const &self) { return DoFSetBasis(self); })
      .def("__deepcopy__",
           [](DoFSetBasis const &self, py::dict) { return DoFSetBasis(self); })
      .def("__repr__",
           [](DoFSetBasis const &self) {
             // Write in Cartesian coordinates
             std::stringstream ss;
             jsonParser json;
             to_json(self, json);
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",  // DoFSetBasis.from_dict
          [](const nlohmann::json &data) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

            jsonParser json{data};
            ParsingDictionary<AnisoValTraits> const *aniso_val_dict = nullptr;
            InputParser<std::vector<DoFSetBasis>> parser(json, aniso_val_dict);
            std::runtime_error error_if_invalid{
                "Error in libcasm.xtal.DoFSetBasis.from_dict"};
            report_and_throw_if_invalid(parser, CASM::log(), error_if_invalid);
            return std::move(*parser.value);
          },
          R"pbdoc(
          Construct a list of DoFSetBasis from a Python dict

          The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/#user-specified-dof-basis>`_
          documents the DoFSetBasis format:

          .. code-block:: Python

              data = {
                  "<dofname>": {
                      "axis_names": ["<axis1_name>", "<axis2_name>", ...],
                      "basis": [
                        [b11, b12, ...],  # axis 1
                        [b21, b22, ...],  # axis 2
                        ...
                      ]
                  }
              }

          Parameters
          ----------
          data: dict
              The Python dict representation of one or more DoFSetBasis

          Returns
          -------
          dof: list[DoFSetBasis]
              The list of DoFSetBasis.

          )pbdoc",
          py::arg("data"))
      .def(
          "to_dict",
          [](DoFSetBasis const &self, py::dict data) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

            jsonParser json;
            to_json(self, json);
            for (auto it = json.begin(); it != json.end(); ++it) {
              data[it.name().c_str()] = nlohmann::json(*it);
            }
            return data;
          },
          R"pbdoc(
          Update a Python dict with the DoFSetBasis

          The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/#user-specified-dof-basis>`_
          documents the DoFSetBasis format.

          .. rubric:: Example usage

          .. code-block:: Python

              # prim: libcasm.xtal.Prim

              global_dof_data = {}
              for dof in prim.global_dof():
                  dof.to_dict(global_dof_data)

              basis_dof_data = []
              for sublat_dof in prim.local_dof():
                  site_dof_data = {}
                  for dof in sublat_dof:
                      dof.to_dict(site_dof_data)
                  basis_dof_data.append(site_dof_data)

          Parameters
          ----------
          data: dict
              A dict to update with the DoFSetBasis.

          Returns
          -------
          data: dict
              The updated dict.

          )pbdoc",
          py::arg("data"));

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
      construction. The :func:`make_primitive` method may be
      used to find the primitive equivalent, and the
      :func:`make_canonical_prim` method may be used to find
      the equivalent with a Niggli cell lattice aligned in a CASM
      standard direction.

      .. rubric:: Special Methods

      - Prim may be copied with :func:`Prim.copy <libcasm.xtal.Prim.copy>`,
        `copy.copy` or `copy.deepcopy`.

      )pbdoc")
      .def(py::init(&make_prim), py::arg("lattice"), py::arg("coordinate_frac"),
           py::arg("occ_dof"),
           py::arg("local_dof") = std::vector<std::vector<DoFSetBasis>>{},
           py::arg("global_dof") = std::vector<DoFSetBasis>{},
           py::arg("occupants") = std::map<std::string, xtal::Molecule>{},
           py::arg("title") = std::string("prim"),
           py::arg("labels") = std::nullopt,
           R"pbdoc(

      .. _prim-init:

      .. rubric:: Constructor

      Parameters
      ----------
      lattice : Lattice
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
      local_dof : list[list[:class:`DoFSetBasis`]], default=[[]]
          Continuous DoF allowed on each basis site. No effect if empty.
          If not empty, the value local_dof[b] is a list of :class:`DoFSetBasis`
          objects describing the DoF allowed on the `b`-th basis site.
      global_dof : list[:class:`DoFSetBasis`], default=[]
          Global continuous DoF allowed for the entire crystal.
      occupants : dict[str,:class:`Occupant`], default=[]
          :class:`Occupant` allowed in the crystal. The keys are names
          used in the occ_dof parameter. This may include isotropic atoms,
          vacancies, atoms with fixed anisotropic properties, and molecular
          occupants. A seperate key and value is required for all species
          with distinct anisotropic properties (i.e. "H2_xy", "H2_xz", and
          "H2_yz" for distinct orientations, or "A.up", and "A.down" for
          distinct collinear magnetic spins, etc.).
      title : str, default="prim"
          A title for the prim. When the prim is used to construct a
          cluster expansion, this must consist of alphanumeric characters
          and underscores only. The first character may not be a number.
      labels : Optional[list[int]] = None
          If provided, an integer for each basis site, greater than or equal to zero,
          that distinguishes otherwise identical sites.
      )pbdoc")
      .def("lattice", &get_prim_lattice, "Returns the lattice, as a copy.")
      .def("coordinate_frac", &get_prim_coordinate_frac,
           "Returns the basis site positions, as columns of a 2d array, in "
           "fractional coordinates with respect to the lattice vectors")
      .def("coordinate_cart", &get_prim_coordinate_cart,
           "Returns the basis site positions, as columns of a 2d array, in "
           "Cartesian coordinates")
      .def("occ_dof", &get_prim_occ_dof,
           "Returns the names of occupants allowed on each basis site")
      .def("local_dof", &get_prim_local_dof,
           "Returns the continuous DoF allowed on each basis site")
      .def(
          "global_dof", &get_prim_global_dof,
          "Returns the continuous DoF allowed for the entire crystal structure")
      .def("occupants", &get_prim_molecules,
           "Returns the :class:`Occupant` allowed in the crystal.")
      .def("labels", &get_prim_labels,
           "Returns the integer label associated with each basis site. If no "
           "labels were provided, it will be a list of -1.")
      .def(
          "copy",
          [](std::shared_ptr<xtal::BasicStructure const> const &prim) {
            return std::make_shared<xtal::BasicStructure const>(*prim);
          },
          R"pbdoc(
           Returns a copy of the Prim.
           )pbdoc")
      .def("__copy__",
           [](std::shared_ptr<xtal::BasicStructure const> const &prim) {
             return std::make_shared<xtal::BasicStructure const>(*prim);
           })
      .def("__deepcopy__",
           [](std::shared_ptr<xtal::BasicStructure const> const &prim,
              py::dict) {
             return std::make_shared<xtal::BasicStructure const>(*prim);
           })
      .def("__repr__",
           [](std::shared_ptr<xtal::BasicStructure const> const &prim) {
             std::stringstream ss;
             jsonParser json;
             COORD_TYPE mode = FRAC;
             bool include_va = false;
             write_prim(*prim, json, mode, include_va);
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",  // Prim.from_dict
          [](const nlohmann::json &data, double xtal_tol) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

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
                By default, basis site positions are written in fractional
                coordinates relative to the lattice vectors. If False, write basis site
                positions in Cartesian coordinates.
            include_va : boolean, default=False
                If a basis site only allows vacancies, it is not printed by default.
                If this is True, basis sites with only vacancies will be included.

            Returns
            -------
            data: dict
                The `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/>`_ documents the expected format.

            )pbdoc")
      .def_static("from_json", &prim_from_json,
                  R"pbdoc(
          Construct a Prim from a JSON-formatted string.

          .. deprecated:: 2.0a8
                Use :func:`Prim.from_dict` instead.

          The
          `Prim reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/BasicStructure/>`_
          documents the expected JSON format.
          )pbdoc",
                  py::arg("prim_json_str"), py::arg("xtal_tol") = TOL)
      .def_static("from_poscar", &prim_from_poscar,
                  R"pbdoc(
            Construct a Prim from a VASP POSCAR file

            Notes
            -----
            If present, selective dynamics are set as :class:`Occupant`
            properties.

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
            prim : Prim
                A Prim

            )pbdoc",
                  py::arg("poscar_path"),
                  py::arg("occ_dof") = std::vector<std::vector<std::string>>{},
                  py::arg("xtal_tol") = TOL)
      .def_static("from_poscar_str", &prim_from_poscar_str,
                  R"pbdoc(
            Construct a Prim from a VASP POSCAR string

            Notes
            -----
            If present, selective dynamics are set as :class:`Occupant`
            properties.

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
            prim : Prim
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
          structure : Structure
               The input structure.

          occ_dof : list[list[str]] = []
              By default, the occupation degrees of freedom (DoF) are
              set to only allow the structure atom types. This may be
              provided, to explicitly set the occupation DoF.

          xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
              Tolerance used for the Prim lattice.

          Returns
          -------
          prim : Prim
                A Prim
          )pbdoc")
      .def("to_json", &prim_to_json, py::arg("frac") = true,
           py::arg("include_va") = false, R"pbdoc(
            Represent the Prim as a JSON-formatted string.

            .. deprecated:: 2.0a8
                Use :func:`Prim.to_dict` instead.

            Parameters
            ----------
            frac : boolean, default=True
                By default, basis site positions are written in fractional coordinates
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
            first : Prim
                First Prim.

            second : Prim
                Second Prim.

            Returns
            -------
            is_same : Prim
                Returns true if Prim are sharing the same data

            )pbdoc");

  m.def("_share_prim", &share_prim, py::arg("init_prim"), R"pbdoc(
            Make a copy of a Prim - sharing same data

            This is for testing purposes.

            Parameters
            ----------
            init_prim : Prim
                Initial prim.

            Returns
            -------
            prim : Prim
                A copy of the initial prim, sharing the same data.

            )pbdoc");

  m.def("_copy_prim", &copy_prim, py::arg("init_prim"), R"pbdoc(
            Make a copy of a Prim - not sharing same data

            This is for testing purposes.

            Parameters
            ----------
            init_prim : Prim
                Initial prim.

            Returns
            -------
            prim : Prim
                A copy of the initial prim, not sharing the same data.

            )pbdoc");

  m.def("make_prim_within", &make_within, py::arg("init_prim"), R"pbdoc(
            Returns an equivalent Prim with all basis site coordinates within the
            unit cell

            Parameters
            ----------
            init_prim : Prim
                The initial prim.

            Returns
            -------
            prim : Prim
                The prim with all basis site coordinates within the unit cell.

            )pbdoc");

  m.def("make_within", &make_within, py::arg("init_prim"),
        "Equivalent to :func:`make_prim_within`");

  m.def("make_primitive_prim", &make_primitive_prim, py::arg("init_prim"),
        R"pbdoc(
            Returns a primitive equivalent Prim

            A :class:`Prim` object is not forced to be the primitive equivalent
            cell at construction. This function finds and returns the primitive
            equivalent cell by checking for internal translations that map all
            basis sites onto equivalent basis sites, including allowed
            occupants and equivalent local degrees of freedom (DoF), if they
            exist.

            Parameters
            ----------
            init_prim : Prim
                The initial prim.

            Returns
            -------
            prim : Prim
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
          init_prim : Prim
              The initial prim.

          Returns
          -------
          prim : Prim
              The prim with canonical lattice.

        )pbdoc");

  m.def("make_canonical", &make_canonical_prim, py::arg("init_prim"),
        "Equivalent to :func:`make_canonical_prim`");

  m.def("asymmetric_unit_indices", &asymmetric_unit_indices, py::arg("prim"),
        R"pbdoc(
          Returns the indices of equivalent basis sites

          Parameters
          ----------
          prim : Prim
              The prim.

          Returns
          -------
          asymmetric_unit_indices : list[list[int]]
              One list of basis site indices for each set of symmetrically equivalent
              basis sites. In other words, the elements of
              ``asymmetric_unit_indices[i]`` are the indices of the `i`-th set of basis
              sites which are symmetrically equivalent to each other.

          )pbdoc");

  m.def("make_prim_factor_group", &make_prim_factor_group, py::arg("prim"),
        R"pbdoc(
          Returns the factor group

          Parameters
          ----------
          prim : Prim
              The prim.

          Returns
          -------
          factor_group : list[:class:`SymOp`]
              The set of symmery operations, with translation lying within the
              primitive unit cell, that leave the lattice vectors, basis site
              coordinates, and all DoF invariant.

          )pbdoc");

  m.def("make_factor_group", &make_prim_factor_group, py::arg("prim"),
        "Equivalent to :func:`make_prim_factor_group`");

  m.def("make_prim_crystal_point_group", &make_prim_crystal_point_group,
        py::arg("prim"),
        R"pbdoc(
          Returns the crystal point group

          Parameters
          ----------
          prim : Prim
              The prim.

          Returns
          -------
          crystal_point_group : list[:class:`SymOp`]
              The crystal point group is the group constructed from the prim factor
              group operations with translation vector set to zero.

          )pbdoc");

  m.def("make_crystal_point_group", &make_prim_crystal_point_group,
        py::arg("prim"), "Equivalent to :func:`make_prim_crystal_point_group`");

  pySymOp
      .def(py::init(&make_symop), py::arg("matrix"), py::arg("translation"),
           py::arg("time_reversal"),
           R"pbdoc(

          .. rubric:: Constructor

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
      .def(
          "matrix_rep",
          [](xtal::SymOp const &op, std::string key) -> Eigen::MatrixXd {
            Eigen::MatrixXd M;
            try {
              AnisoValTraits traits(key);
              M = traits.symop_to_matrix(get_matrix(op), get_translation(op),
                                         get_time_reversal(op));
            } catch (std::exception &e) {
              std::stringstream msg;
              msg << "Error getting matrix rep: CASM does not know how to "
                     "transform the property '"
                  << key << "'.";
              throw std::runtime_error(msg.str());
            }
            return M;
          },
          R"pbdoc(
          Returns the matrix representation of a symmetry operation for transforming \
          properties

          Parameters
          ----------
          key: str
              The name of the CASM-supported property to be transformed.

          Returns
          -------
          matrix_rep : numpy.ndarray[numpy.float64[m, m]]
              The matrix representation for transforming properties. The matrix is
              square, with dimension equal to the standard dimension of the specified
              property. For example, `m=3` for `key="disp"`, and `m=6` for
              `key="Hstrain"`. Local properties, such as `"disp"`, stored as columns of
              array `local_values`, can then be transformed using
              ``matrix_rep @ local_values``. Global properties, such as `"Hstrain"`,
              stored as array `global_values` with a single column, can similarly be
              transformed using  ``matrix_rep @ global_values``.

          )pbdoc",
          py::arg("key"))
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
          "Transform Cartesian coordinates, represented as a 1d array",
          py::is_operator())
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
          "a matrix.",
          py::is_operator())
      .def(
          "__mul__",
          [](xtal::SymOp const &lhs, xtal::SymOp const &rhs) {
            return lhs * rhs;
          },
          py::arg("rhs"),
          "Construct the SymOp equivalent to applying first `rhs`, then this.",
          py::is_operator())
      .def(
          "__mul__",
          [](xtal::SymOp const &op,
             std::map<std::string, Eigen::MatrixXd> const &properties) {
            return copy_apply(op, properties);
          },
          py::arg("rhs"),
          "Transform CASM-supported properties (local or global).",
          py::is_operator())
      .def(
          "__mul__",
          [](xtal::SymOp const &op, xtal::Lattice const &lattice) {
            return sym::copy_apply(op, lattice);
          },
          py::arg("lattice"), "Transform a Lattice.", py::is_operator())
      .def(
          "__mul__",
          [](xtal::SymOp const &op, xtal::SimpleStructure const &simple) {
            return copy_apply(op, simple);
          },
          py::arg("structure"), "Transform a Structure.", py::is_operator())
      .def(
          "copy", [](xtal::SymOp const &self) { return xtal::SymOp(self); },
          R"pbdoc(
           Returns a copy of the SymOp.
           )pbdoc")
      .def("__copy__",
           [](xtal::SymOp const &self) { return xtal::SymOp(self); })
      .def("__deepcopy__",
           [](xtal::SymOp const &self, py::dict) { return xtal::SymOp(self); })
      .def("__repr__",
           [](xtal::SymOp const &op) {
             std::stringstream ss;
             jsonParser json;
             json["matrix"] = xtal::get_matrix(op);
             to_json_array(xtal::get_translation(op), json["tau"]);
             json["time_reversal"] = xtal::get_time_reversal(op);
             ss << json;
             return ss.str();
           })
      .def_static(
          "from_dict",  // SymOp.from_dict
          [](const nlohmann::json &data) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

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

          .. rubric:: Constructor

          Parameters
          ----------
          op : SymOp
              The symmetry operation.
          lattice : Lattice
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
              - the rotation axis of inversion * self, if this is an improper rotation
                (then the axis is a normal vector for a mirror plane)
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
              - the rotation angle of inversion * self, if this is an improper rotation
                (then the axis is a normal vector for a mirror plane)
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

              The `screw_glide_shift` vector is in Cartesian coordinates.
          )pbdoc")
      .def("location", &get_syminfo_location, R"pbdoc(
          A Cartesian coordinate that is invariant to the operation (if one exists)

          Returns
          -------
          location: numpy.ndarray[numpy.float64[3, 1]]
              The location is in Cartesian coordinates. This does not exist for the
              identity operation.
          )pbdoc")
      .def("brief_cart", &get_syminfo_brief_cart, R"pbdoc(
          A brief description of the symmetry operation, in Cartesian coordinates

          Returns
          -------
          brief_cart: str
              A brief string description of the symmetry operation, in Cartesian
              coordinates, following the conventions of (International Tables for
              Crystallography (2015). Vol. A. ch. 1.4, pp. 50-59).
          )pbdoc")
      .def("brief_frac", &get_syminfo_brief_frac, R"pbdoc(
          A brief description of the symmetry operation, in fractional coordinates

          Returns
          -------
          brief_cart: str
              A brief string description of the symmetry operation, in fractional
              coordinates, following the conventions of (International Tables for
              Crystallography (2015). Vol. A. ch. 1.4, pp. 50-59).
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

          .. deprecated:: 2.0a8
                Use :func:`SymInfo.to_dict` instead.

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

    .. rubric:: Constructor

    Parameters
    ----------
    lattice : Lattice
        The Lattice. Note: The lattice tolerance is not saved in Structure.
    atom_coordinate_frac : array_like, shape (3, n)
        Atom positions, as columns of a matrix, in fractional
        coordinates with respect to the lattice vectors.
    atom_type : list[str], size=n
        Atom type names.
    atom_properties : dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with individual atoms, if present. Keys must
        be the name of a CASM-supported property type. Values are arrays with
        dimensions matching the standard dimension of the property type.
    mol_coordinate_frac : array_like, shape (3, n)
        Molecule positions, as columns of a matrix, in fractional
        coordinates with respect to the lattice vectors.
    mol_type : list[str], size=n
        Molecule type names.
    mol_properties : dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with individual molecules, if present. Keys
        must be the name of a CASM-supported property type. Values are arrays with
        dimensions matching the standard dimension of the property type.
    global_properties : dict[str,  numpy.ndarray[numpy.float64[m, n]]], default={}
        Continuous properties associated with entire crystal, if present. Keys must
        be the name of a CASM-supported property type. Values are (m, 1) arrays with
        dimensions matching the standard dimension of the property type.
    )pbdoc")
      .def("lattice", &get_simplestructure_lattice, R"pbdoc(
            Returns the lattice, as a copy

            Parameters
            ----------
            xtal_tol: float = :data:`~libcasm.casmglobal.TOL`
                Tolerance used for lattice.

            Returns
            -------
            lattice : Lattice
                The lattice, returned as a copy.

            )pbdoc",
           py::arg("xtal_tol") = TOL)
      .def("atom_coordinate_cart", &get_simplestructure_atom_coordinate_cart,
           "Returns the atom positions, as columns of a 2d array, in Cartesian "
           "coordinates.")
      .def(
          "atom_coordinate_frac", &get_simplestructure_atom_coordinate_frac,
          "Returns the atom positions, as columns of a 2d array, in fractional "
          "coordinates with respect to the lattice vectors.")
      .def("atom_type", &get_simplestructure_atom_type,
           "Returns a list with atom type names.")
      .def("atom_properties", &get_simplestructure_atom_properties,
           "Returns continuous properties associated with individual atoms, if "
           "present.")
      .def("mol_coordinate_cart", &get_simplestructure_mol_coordinate_cart,
           "Returns the molecule positions, as columns of a 2d array, in "
           "Cartesian coordinates.")
      .def("mol_coordinate_frac", &get_simplestructure_mol_coordinate_frac,
           "Returns the molecule positions, as columns of a 2d array, in "
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
          "from_dict",  // Structure.from_dict
          [](const nlohmann::json &data) {
            // print errors and warnings to sys.stdout
            py::scoped_ostream_redirect redirect;

            jsonParser json{data};
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
          [](xtal::SimpleStructure const &simple,
             std::vector<std::string> const &excluded_species, bool frac) {
            jsonParser json;
            COORD_TYPE mode = frac ? FRAC : CART;
            std::set<std::string> _excluded_species(excluded_species.begin(),
                                                    excluded_species.end());
            to_json(simple, json, _excluded_species, mode);
            return static_cast<nlohmann::json>(json);
          },
          py::arg("excluded_species") =
              std::vector<std::string>({"Va", "VA", "va"}),
          py::arg("frac") = true, R"pbdoc(
          Represent the Structure as a Python dict.

          Parameters
          ----------
          excluded_species : list[str] = ["Va", "VA", "va"]
              The names of any molecular or atomic species that should not be included
              in the output.
          frac : boolean, default=True
              By default, coordinates are written in fractional coordinates relative to
              the lattice vectors. If False, write coordinates in Cartesian coordinates.

          Returns
          -------
          data: dict
              The `Structure reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/SimpleStructure/>`_ documents the format.
          )pbdoc")
      .def("__repr__",
           [](xtal::SimpleStructure const &simple) {
             std::stringstream ss;
             jsonParser json;
             COORD_TYPE mode = FRAC;
             std::set<std::string> excluded_species = {"Va", "VA", "va"};
             to_json(simple, json, excluded_species, mode);
             ss << json;
             return ss.str();
           })
      .def_static("from_json", &simplestructure_from_json, R"pbdoc(
          Construct a Structure from a JSON-formatted string.

          .. deprecated:: 2.0a8
                Use :func:`Structure.from_dict` instead.

          The
          `Structure reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/SimpleStructure/>`_
          documents the expected JSON format.
          )pbdoc",
                  py::arg("structure_json_str"))
      .def_static("from_poscar", &simplestructure_from_poscar,
                  R"pbdoc(
            Construct a Structure from a VASP POSCAR file

            Parameters
            ----------
            poscar_path : str
                Path to the POSCAR file
            mode : str = "atoms"
                Read POSCAR and construct a Structure with atom or molecule types,
                coordinates, and if present, "selectivedynamics" properties. Accepts
                one of "atoms", "molecules", or "both".

            Returns
            -------
            structure : Structure
                A Structure, with lattice, types, coordinates, and if present,
                "selectivedynamics" properties.

            )pbdoc",
                  py::arg("poscar_path"),
                  py::arg("mode") = std::string("atoms"))
      .def_static("from_poscar_str", &simplestructure_from_poscar_str, R"pbdoc(
            Construct a Structure from a VASP POSCAR string

            Parameters
            ----------
            poscar_str : str
                The POSCAR as a string
            mode : str = "atoms"
                Read POSCAR and construct a Structure with atom or molecule types,
                coordinates, and if present, "selectivedynamics" properties. Accepts
                one of "atoms", "molecules", or "both".

            Returns
            -------
            structure : Structure
                A Structure, with lattice, atom_type, and atom coordinates, and
                if present, "selectivedynamics" atom properties.

            )pbdoc",
                  py::arg("poscar_str"), py::arg("mode") = std::string("atoms"))
      .def("to_json", &simplestructure_to_json,
           R"pbdoc(
          Represent the Structure as a JSON-formatted string.

          .. deprecated:: 2.0a8
              Use :func:`Structure.to_dict` instead.

          The `Structure reference <https://prisms-center.github.io/CASMcode_docs/formats/casm/crystallography/SimpleStructure/>`_
          documents the expected JSON format.
          ")pbdoc")
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
            structure : Structure
                A Structure

            )pbdoc",
          py::arg("sort") = true, py::arg("title") = "<title>",
          py::arg("ignore") = std::vector<std::string>{"VA", "Va", "va"},
          py::arg("cart_coordinate_mode") = false)
      .def(
          "copy",
          [](xtal::SimpleStructure const &self) {
            return xtal::SimpleStructure(self);
          },
          R"pbdoc(
           Returns a copy of the Structure.
           )pbdoc")
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
              structure2 : Structure
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

        Notes
        -----
        This method only considers atom coordinates and types. Molecular coordinates
        and types are not considered. Properties are not considered. The default CASM
        tolerance is used for comparisons. To consider molecules or properties, or to
        use a different tolerance, use a :class:`Prim` with :class:`Occupant` that have
        properties.

        Parameters
        ----------
        structure : Structure
            The structure.


        Returns
        -------
        factor_group : list[SymOp]
            The the set of symmery operations, with translation lying within the
            primitive unit cell, that leave the lattice vectors, atom coordinates,
            and atom types invariant.

        )pbdoc");

  m.def("make_factor_group", &make_simplestructure_factor_group,
        py::arg("structure"),
        "Equivalent to :func:`make_structure_factor_group`");

  m.def("make_structure_crystal_point_group",
        &make_simplestructure_crystal_point_group, py::arg("structure"),
        R"pbdoc(
           Returns the crystal point group of an atomic structure

           Parameters
           ----------
           structure : Structure
               The structure.

           Returns
           -------
           crystal_point_group : list[SymOp]
               The crystal point group is the group constructed from the structure
               factor group operations with translation vector set to zero.

           Notes
           -----
           Currently this method only considers atom coordinates and types. Molecular
           coordinates and types are not considered. Properties are not considered.
           The default CASM tolerance is used for comparisons. To consider molecules
           or properties, or to use a different tolerance, use a Prim.
           )pbdoc");

  m.def("make_crystal_point_group", &make_simplestructure_crystal_point_group,
        py::arg("structure"),
        "Equivalent to :func:`make_structure_crystal_point_group`");

  m.def("make_structure_within", &make_simplestructure_within,
        py::arg("init_structure"), R"pbdoc(
            Returns an equivalent Structure with all atom and mol site coordinates
            within the unit cell

            Parameters
            ----------
            init_structure : Structure
                The initial structure.

            Returns
            -------
            structure : Structure
                The structure with all atom and mol site coordinates within the unit
                cell.

            )pbdoc");

  m.def("make_within", &make_simplestructure_within, py::arg("init_structure"),
        "Equivalent to :func:`make_structure_within`");

  m.def("make_primitive_structure", &make_primitive_simplestructure,
        py::arg("init_structure"), R"pbdoc(
        Returns a primitive equivalent atomic Structure

        This function finds and returns the primitive equivalent cell by checking for
        internal translations that map all atoms onto equivalent atoms.

        Notes
        -----
        Currently this method only considers atom coordinates and types. Molecular
        coordinates and types are not considered. Properties are not considered.
        The default CASM tolerance is used for comparisons. To consider molecules
        or properties, or to use a different tolerance, use a Prim.

        Parameters
        ----------
        init_structure: _xtal.Structure
            The initial Structure

        Returns
        -------
        structure: _xtal.Structure
            The primitive equivalent Structure
        )pbdoc");

  m.def("make_canonical_structure", &make_canonical_simplestructure,
        py::arg("init_structure"), R"pbdoc(
        Returns an equivalent Structure with canonical lattice

        Finds the canonical right-handed Niggli cell of the lattice, applying
        lattice point group operations to find the equivalent lattice in a
        standardized orientation. The canonical orientation prefers lattice
        vectors that form symmetric matrices with large positive values on the
        diagonal and small values off the diagonal. See also `Lattice Canonical Form`_.

        .. _`Lattice Canonical Form`: https://prisms-center.github.io/CASMcode_docs/formats/lattice_canonical_form/

        Parameters
        ----------
        init_structure: _xtal.Structure
            The initial Structure

        Returns
        -------
        structure: _xtal.Structure
            The structure with canonical lattice.

        )pbdoc");

  m.def("make_superstructure", &make_superstructure,
        py::arg("transformation_matrix_to_super").noconvert(),
        py::arg("structure"),
        R"pbdoc(
      Make a superstructure

      Parameters
      ----------
      transformation_matrix_to_super: numpy.ndarray[numpy.int64[3, 3]]
          The transformation matrix, `T`, relating the superstructure lattice vectors,
          `S`, to the unit structure lattice vectors, `L`, according to ``S = L @ T``,
          where `S` and `L` are shape=(3,3) matrices with lattice vectors as columns.
      structure: Structure
          The unit structure used to form the superstructure.

      Returns
      -------
      superstructure: Structure
          The superstructure.
      )pbdoc");

  m.def("make_equivalent_property_values", &make_equivalent_property_values,
        py::arg("point_group"), py::arg("x"), py::arg("property_type"),
        py::arg("basis") = Eigen::MatrixXd(0, 0), py::arg("tol") = TOL,
        R"pbdoc(
      Make the set of symmetry equivalent property values

      Parameters
      ----------
      point_group : list[:class:`SymOp`]
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

    For more information on strain metrics and using a symmetry-adapted or
    user-specified basis, see :ref:`Strain DoF <sec-strain-dof>`.

    :class:`StrainConverter` supports the following choices of symmetric
    strain metrics, :math:`E`, shape=(3,3):

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

    .. rubric:: Constructor

    Parameters
    ----------
    metric: str (optional, default='Ustrain')
        Choice of strain metric, one of: 'Ustrain', 'GLstrain', 'Hstrain',
        'EAstrain', 'Bstrain'

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
           Q: numpy.ndarray[numpy.float64[3,3]]
               The shape=(3,3) isometry matrix, :math:`Q`, of the
               deformation tensor.
           U: numpy.ndarray[numpy.float64[3,3]]
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
            Q: numpy.ndarray[numpy.float64[3,3]]
                The shape=(3,3) isometry matrix, :math:`Q`, of the
                deformation tensor.
            V: numpy.ndarray[numpy.float64[3,3]]
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
           E_vector_in_standard_basis: numpy.ndarray[numpy.float64[6,1]]
               Strain metric vector, expressed in the standard basis. This is
               equivalent to `basis @ E_vector`.
           )pbdoc")
      .def("from_standard_basis", &xtal::StrainConverter::from_standard_basis,
           py::arg("E_vector_in_standard_basis"),
           R"pbdoc(
           Convert strain metric vector from standard basis to converter basis.

           Parameters
           ----------
           E_vector_in_standard_basis: array_like, shape=(6,1)
               Strain metric vector, expressed in the standard basis. This is
               equivalent to `basis @ E_vector`.

           Returns
           -------
           E_vector: numpy.ndarray[numpy.float64[dim,1]]
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
           E_matrix: numpy.ndarray[numpy.float64[3, 3]]
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
           E_vector: numpy.ndarray[numpy.float64[dim,1]]
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

      which decomposes strain space into irreducible subspaces (subspaces which do not
      mix under application of symmetry).

      For more information on strain metrics and the symmetry-adapted strain basis,
      see :ref:`Strain DoF <sec-strain-dof>`.

      Returns
      -------
      symmetry_adapted_strain_basis: numpy.ndarray[numpy.float64[6, 6]]
          The symmetry-adapted strain basis, :math:`B^{\vec{e}}`.
      )pbdoc");

  // IntegralSiteCoordinate -- declaration
  py::class_<xtal::UnitCellCoord> pyIntegralSiteCoordinate(
      m, "IntegralSiteCoordinate", R"pbdoc(
      Specify a site using integer sublattice and unit cell indices

      .. rubric:: Special Methods

      Translate an :class:`IntegralSiteCoordinate` using operators
      ``+``, ``-``, ``+=``, ``-=``:

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


      Sort :class:`IntegralSiteCoordinate` by lexicographical order of
      unit cell indices :math:`(i,j,k)` then sublattice index `b` using ``<``, ``<=``,
      ``>``, ``>=``, and compare using ``==``, ``!=``:

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

      Represent :class:`IntegralSiteCoordinate` as the string
      ``"b, i j k"``, where `b` is the sublattice index and `i j k` are the unit cell
      indices, using ``str()``:

      .. code-block:: Python

          import numpy as np
          from libcasm.xtal import IntegralSiteCoordinate

          # construct IntegralSiteCoordinate
          site = IntegralSiteCoordinate(0, np.array([1, 2, 3]))

          assert str(site) == "0, 1 2 3"

      - IntegralSiteCoordinate may be copied with
        :func:`IntegralSiteCoordinate.copy <libcasm.xtal.IntegralSiteCoordinate.copy>`,
        `copy.copy`, or `copy.deepcopy`.

      )pbdoc");

  // IntegralSiteCoordinateRep -- declaration
  py::class_<xtal::UnitCellCoordRep> pyIntegralSiteCoordinateRep(
      m, "IntegralSiteCoordinateRep", R"pbdoc(
      Symmetry representation for transforming IntegralSiteCoordinate

      An :class:`IntegralSiteCoordinateRep` is a symmetry representation for
      transforming :class:`IntegralSiteCoordinate` objects equivalent to the
      action of a :class:`SymOp` using only integer operations with:

      .. math::

          b^{\ after} = p_{b^{\ before}}

      .. math::

          \vec{u}^{\ after} = \mathbf{P} \vec{u}^{\ before} + \vec{t}_{b^{\ before}},

      where :math:`\vec{p}` is a permutation that describes how the sublattice
      index `b` transforms, and :math:`\mathbf{P}` is an 3x3 integer matrix and
      :math:`\vec{t}_{b^{\ before}}` is a fractional coordinate translation
      vector (which depends on the initial sublattice) describing the change in
      the fractional coordinate :math:`\vec{u} = [i,j,k]` after application of
      symmetry.

      .. rubric:: Special Methods

      Copy and transform an :class:`IntegralSiteCoordinate` via multiplication
      operator ``*`` or the :func:`copy_apply` method; or apply in-place with
      the :func:`apply` method:

      .. code-block:: Python

          from libcasm.xtal import IntegralSiteCoordinate, IntegralSiteCoordinateRep
          rep = IntegralSiteCoordinateRep(...)
          site = IntegralSiteCoordinate(...)

          # copy and transform using `*` or `copy_apply`
          transformed_site_1 = rep * site
          transformed_site_2 = copy_apply(rep, site)
          assert transformed_site_1 == transformed_site_2

          # or copy, then transform in-place using `apply`
          copied_site = site.copy()
          apply(rep, copied_site)
          assert copied_site == transformed_site_1

      - IntegralSiteCoordinateRep may be copied with
        :func:`IntegralSiteCoordinateRep.copy <libcasm.xtal.IntegralSiteCoordinateRep.copy>`,
        `copy.copy`, or `copy.deepcopy`.

      )pbdoc");

  // IntegralSiteCoordinate -- definition
  pyIntegralSiteCoordinate
      .def(py::init(&make_integral_site_coordinate),
           "Construct an IntegralSiteCoordinate", py::arg("sublattice"),
           py::arg("unitcell"), R"pbdoc(

      .. rubric:: Constructor

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
          "raised if it is not possible with the given tolerance.")
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
          "raised if it is not possible with the given tolerance.")
      .def("sublattice", &xtal::UnitCellCoord::sublattice,
           "Returns the sublattice index, `b`.")
      .def(
          "unitcell",
          [](xtal::UnitCellCoord const &self) {
            return static_cast<Eigen::Vector3l>(self.unitcell());
          },
          "Returns the unit cell indices, :math:`(i,j,k)`, as a shape=(3,) "
          "integer array.")
      .def(
          "__repr__",
          [](xtal::UnitCellCoord const &self) {
            std::stringstream ss;
            jsonParser json;
            to_json(self, json);
            ss << json;
            return ss.str();
          },
          "Represent IntegralSiteCoordinate as `[b, i, j, k]`, where `b` is "
          "the sublattice index and `i, j, k` are the unit cell coordinates.")
      .def(
          "to_list",
          [](xtal::UnitCellCoord const &self) {
            std::vector<Index> list;
            for (int i = 0; i < 4; ++i) {
              list.push_back(self[i]);
            }
            return list;
          },
          "Represent IntegralSiteCoordinate as a list ``[b, i, j, k]``.")
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
          "Construct IntegralSiteCoordinate from a list ``[b, i, j, k]``.")
      .def(
          "__iadd__",
          [](xtal::UnitCellCoord &self, Eigen::Vector3l const &translation) {
            self += xtal::UnitCell(translation);
            return self;
          },
          py::arg("translation").noconvert(),
          "Translates the integral site coordinate by adding unit cell indices")
      .def(
          "__add__",
          [](xtal::UnitCellCoord const &self,
             Eigen::Vector3l const &translation) {
            return self + xtal::UnitCell(translation);
          },
          py::arg("translation").noconvert(),
          "Translates the integral site coordinate by adding unit cell indices")
      .def(
          "__isub__",
          [](xtal::UnitCellCoord &self, Eigen::Vector3l const &translation) {
            self -= xtal::UnitCell(translation);
            return self;
          },
          py::arg("translation").noconvert(),
          "Translates the integral site coordinate by subtracting unit cell "
          "indices")
      .def(
          "__sub__",
          [](xtal::UnitCellCoord const &self,
             Eigen::Vector3l const &translation) {
            return self - xtal::UnitCell(translation);
          },
          py::arg("translation").noconvert(),
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
           "Sorts coordinates by lexicographical order of :math:`(i,j,k)` then "
           "`b`")
      .def(py::self <= py::self,
           "Sorts coordinates by lexicographical order of :math:`(i,j,k)` then "
           "`b`")
      .def(py::self > py::self,
           "Sorts coordinates by lexicographical order of :math:`(i,j,k)` then "
           "`b`")
      .def(py::self >= py::self,
           "Sorts coordinates by lexicographical order of :math:`(i,j,k)` then "
           "`b`")
      .def(py::self == py::self, "True if coordinates are equal")
      .def(py::self != py::self, "True if coordinates are not equal")
      .def(
          "copy",
          [](xtal::UnitCellCoord const &self) {
            return xtal::UnitCellCoord(self);
          },
          R"pbdoc(
           Returns a copy of the IntegralSiteCoordinate.
           )pbdoc")
      .def("__copy__",
           [](xtal::UnitCellCoord const &self) {
             return xtal::UnitCellCoord(self);
           })
      .def("__deepcopy__", [](xtal::UnitCellCoord const &self, py::dict) {
        return xtal::UnitCellCoord(self);
      });

  // IntegralSiteCoordinateRep -- definition
  pyIntegralSiteCoordinateRep
      .def(py::init(&CASMpy::make_unitcellcoord_rep),
           "Construct an IntegralSiteCoordinateRep", py::arg("op"),
           py::arg("prim"), R"pbdoc(

      .. rubric:: Constructor

      Parameters
      ----------
      op : SymOp
          The symmetry operation.
      prim : Prim
          The prim defining IntegralSiteCoordinate that will be transformed.
      )pbdoc")
      .def(
          "__mul__",
          [](xtal::UnitCellCoordRep const &rep,
             xtal::UnitCellCoord const &integral_site_coordinate) {
            return copy_apply(rep, integral_site_coordinate);
          },
          py::arg("integral_site_coordinate"),
          "Transform an :class:`IntegralSiteCoordinate`", py::is_operator())
      .def(
          "copy",
          [](xtal::UnitCellCoordRep const &self) {
            return xtal::UnitCellCoordRep(self);
          },
          R"pbdoc(
           Returns a copy of the IntegralSiteCoordinate.
           )pbdoc")
      .def("__copy__",
           [](xtal::UnitCellCoordRep const &self) {
             return xtal::UnitCellCoordRep(self);
           })
      .def("__deepcopy__",
           [](xtal::UnitCellCoordRep const &self, py::dict) {
             return xtal::UnitCellCoordRep(self);
           })
      .def("__repr__", [](xtal::UnitCellCoordRep const &self) {
        std::stringstream ss;
        jsonParser json;
        json["sublattice_after"] = self.sublattice_index;
        json["matrix_frac"] = self.point_matrix;
        json["tau_frac"] = jsonParser::array();
        for (auto const &tau_frac : self.unitcell_indices) {
          jsonParser tjson;
          to_json(tau_frac, tjson, jsonParser::as_array());
          json["tau_frac"].push_back(tjson);
        }
        ss << json;
        return ss.str();
      });

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
      Convert between integral site indices :math:`(b,i,j,k)` and linear site
      index :math:`l`.
      )pbdoc")
      .def(py::init(&make_SiteIndexConverter),
           py::arg("transformation_matrix_to_super").noconvert(),
           py::arg("n_sublattice"),
           R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
              The transformation matrix, `T`, relating the superstructure lattice
              vectors, `S`, to the unit structure lattice vectors, `L`, according to
              ``S = L @ T``, where `S` and `L` are shape=(3,3)  matrices with lattice
              vectors as columns.

          n_sublattice: int
              The number of sublattices in the :class:`Prim`.

          )pbdoc")
      .def("never_bring_within",
           &xtal::UnitCellCoordIndexConverter::never_bring_within,
           R"pbdoc(
            Prevent the index converter from bringing
            :class:`IntegralSiteCoordinate` within the supercell when
            querying for the index.
          )pbdoc")
      .def("always_bring_within",
           &xtal::UnitCellCoordIndexConverter::always_bring_within,
           R"pbdoc(
            Automatically bring :class:`IntegralSiteCoordinate` values
            within the supercell when querying for the index (on by default).
          )pbdoc")
      .def("bring_within", &xtal::UnitCellCoordIndexConverter::bring_within,
           R"pbdoc(
          Bring the given :class:`IntegralSiteCoordinate` into the
          superlattice using superlattice translations.
          )pbdoc",
           py::arg("integral_site_coordinate"))
      .def(
          "linear_site_index",
          [](xtal::UnitCellCoordIndexConverter const &f,
             xtal::UnitCellCoord const &bijk) -> Index { return f(bijk); },
          R"pbdoc(
          Given the :class:`IntegralSiteCoordinate` of a site, retrieve its
          corresponding linear index. By default, if
          :func:`SiteIndexConverter.never_bring_within` has not been
          called, the :class:`IntegralSiteCoordinate` is brought within
          the superlattice using superlattice translations.
          )pbdoc",
          py::arg("integral_site_coordinate"))
      .def(
          "integral_site_coordinate",
          [](xtal::UnitCellCoordIndexConverter const &f,
             Index const &linear_site_index) -> xtal::UnitCellCoord {
            return f(linear_site_index);
          },
          R"pbdoc(
          Given the linear index of a site, retrieve the corresponding
          :class:`IntegralSiteCoordinate`.
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
      .def(py::init(&make_UnitCellIndexConverter),
           py::arg("transformation_matrix_to_super").noconvert(),
           R"pbdoc(

          .. rubric:: Constructor

          Parameters
          ----------
          transformation_matrix_to_super: array_like, shape=(3,3), dtype=int
              The transformation matrix, T, relating the superstructure lattice
              vectors, S, to the unit structure lattice vectors, L, according to
              S = L @ T, where S and L are shape=(3,3)  matrices with lattice vectors
              as columns.

          )pbdoc")
      .def(
          "never_bring_within",
          // &xtal::UnitCellIndexConverter::never_bring_within,
          [](xtal::UnitCellIndexConverter &f) { f.never_bring_within(); },
          R"pbdoc(
            Prevent the index converter from bringing unit cell indices :math:`(i,j,k)`
            within the supercell when querying for the index.
          )pbdoc")
      .def(
          "always_bring_within",
          // &xtal::UnitCellIndexConverter::always_bring_within,
          [](xtal::UnitCellIndexConverter &f) { f.always_bring_within(); },
          R"pbdoc(
            Automatically bring unit cell indices :math:`(i,j,k)` within the supercell
            when querying for the index (on by default).
          )pbdoc")
      .def(
          "bring_within",
          //&xtal::UnitCellIndexConverter::bring_within,
          [](xtal::UnitCellIndexConverter &f, Eigen::Vector3l const &unitcell) {
            return f.bring_within(unitcell);
          },
          R"pbdoc(
          Bring the given :class:`IntegralSiteCoordinate` into the
          superlattice using superlattice translations.
          )pbdoc",
          py::arg("unitcell").noconvert())
      .def(
          "linear_unitcell_index",
          [](xtal::UnitCellIndexConverter const &f,
             Eigen::Vector3l const &unitcell) -> Index { return f(unitcell); },
          R"pbdoc(
          Given unitcell indices, :math:`(i,j,k)`, retrieve the corresponding linear
          unitcell index. By default, if
          :func:`IntegralSiteCoordinateConverter.never_bring_within` has
          not been called, the lattice point is brought within the superlattice using
          superlattice translations.
          )pbdoc",
          py::arg("unitcell").noconvert())
      .def(
          "unitcell",
          [](xtal::UnitCellIndexConverter const &f,
             Index const &linear_unitcell_index) -> Eigen::Vector3l {
            return f(linear_unitcell_index);
          },
          R"pbdoc(
          Given the linear unitcell index, retrieve the corresponding unitcell indices
          :math:`(i,j,k)`.
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
