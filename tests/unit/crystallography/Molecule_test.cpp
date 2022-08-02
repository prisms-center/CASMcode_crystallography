#include "gtest/gtest.h"

/// What is being tested:
#include "casm/crystallography/Molecule.hh"

/// What is being used to test it:
#include "casm/misc/CASM_Eigen_math.hh"

using namespace CASM;
using namespace xtal;

TEST(MoleculeTest, AtomPositionTest1) {
  Eigen::Vector3d vec(0.0, 0.2, 0.4);
  double tol(1e-5);

  AtomPosition atom_pos(vec, "A");

  EXPECT_EQ(atom_pos.name(), "A");
  EXPECT_EQ(almost_equal(atom_pos.cart(), vec, tol), true);
}

TEST(MoleculteTest, MoleculeTest1) {
  Eigen::Vector3d vec(0.0, 0.2, 0.4);
  double tol(1e-5);
  bool divisible(true);

  Molecule mol_a = Molecule::make_atom("A");
  Molecule mol_h2o = Molecule(
      "H2O",
      {AtomPosition(vec, "H"), AtomPosition(vec, "H"), AtomPosition(vec, "O")});
  Molecule mol_Zr2 =
      Molecule("Zr2", {AtomPosition(vec, "Zr"), AtomPosition(-1.0 * vec, "Zr")},
               divisible);
  Molecule mol_va = Molecule::make_vacancy();
  Molecule mol_va2 =
      Molecule::make_atom("Va");  // will make vacancy with no atoms

  EXPECT_EQ(mol_a.size(), 1);
  EXPECT_EQ(mol_a.name(), "A");
  EXPECT_EQ(mol_a.is_vacancy(), false);
  EXPECT_EQ(mol_a.atom(0).name() == "A", true);

  EXPECT_EQ(mol_h2o.size(), 3);
  EXPECT_EQ(mol_h2o.name(), "H2O");
  EXPECT_EQ(mol_h2o.is_vacancy(), false);
  EXPECT_EQ(mol_h2o.atoms().size(), 3);
  EXPECT_EQ(mol_h2o.atoms().at(0).name(), "H");
  EXPECT_EQ(mol_h2o.atoms().at(1).name(), "H");
  EXPECT_EQ(mol_h2o.atoms().at(2).name(), "O");
  EXPECT_EQ(mol_h2o.atom(0).name(), "H");
  EXPECT_EQ(mol_h2o.atom(1).name(), "H");
  EXPECT_EQ(mol_h2o.atom(2).name(), "O");
  EXPECT_EQ(mol_h2o.is_divisible(), false);
  EXPECT_EQ(mol_h2o.is_indivisible(), true);

  EXPECT_EQ(mol_Zr2.size(), 2);
  EXPECT_EQ(mol_Zr2.name(), "Zr2");
  EXPECT_EQ(mol_Zr2.is_vacancy(), false);
  EXPECT_EQ(mol_Zr2.is_divisible(), true);
  EXPECT_EQ(mol_Zr2.is_indivisible(), false);

  // EXPECT_EQ(mol_va.size(), 0); // if Molecule empty for Va
  EXPECT_EQ(mol_va.size(), 1);
  EXPECT_EQ(mol_va.name(), "Va");
  EXPECT_EQ(mol_va.is_vacancy(), true);

  // EXPECT_EQ(mol_va2.size(), 0); // if Molecule empty for Va
  EXPECT_EQ(mol_va2.size(), 1);
  EXPECT_EQ(mol_va2.name(), "Va");
  EXPECT_EQ(mol_va2.is_vacancy(), true);
}

TEST(MoleculeTest, IdenticalTest1) {
  Eigen::Vector3d pos0(0.0, 0.0, 0.4);
  Eigen::Vector3d pos1(0.0, 0.0, -0.4);
  double tol(1e-5);

  Molecule mol_a2_i =
      Molecule("A2", {AtomPosition(pos0, "A"), AtomPosition(pos1, "A")});

  Molecule mol_a2_ii =
      Molecule("A2", {AtomPosition(pos1, "A"), AtomPosition(pos0, "A")});

  Molecule mol_ab_i =
      Molecule("AB", {AtomPosition(pos0, "A"), AtomPosition(pos1, "B")});

  {
    Permutation atom_position_perm(mol_a2_i.size());
    bool is_identical = mol_a2_i.identical(mol_a2_i, tol, atom_position_perm);
    EXPECT_TRUE(is_identical);
    EXPECT_EQ(atom_position_perm[0], 0);
    EXPECT_EQ(atom_position_perm[1], 1);
  }

  {
    Permutation atom_position_perm(mol_a2_i.size());
    bool is_identical = mol_a2_i.identical(mol_a2_ii, tol, atom_position_perm);
    EXPECT_TRUE(is_identical);
    EXPECT_EQ(atom_position_perm[0], 1);
    EXPECT_EQ(atom_position_perm[1], 0);
  }

  {
    Permutation atom_position_perm(mol_a2_i.size());
    bool is_identical = mol_a2_i.identical(mol_ab_i, tol, atom_position_perm);
    EXPECT_FALSE(is_identical);
  }
}
