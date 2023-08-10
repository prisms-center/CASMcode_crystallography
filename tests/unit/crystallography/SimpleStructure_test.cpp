#include "casm/crystallography/SimpleStructure.hh"

#include "autotools.hh"
#include "casm/casm_io/json/jsonParser.hh"
#include "casm/crystallography/SimpleStructureTools.hh"
#include "casm/crystallography/SymType.hh"
#include "casm/crystallography/io/SimpleStructureIO.hh"
#include "casm/misc/CASM_Eigen_math.hh"
#include "gtest/gtest.h"
#include "testdir.hh"

using namespace CASM;

TEST(SimpleStructure, Test1) {
  xtal::SimpleStructure structure;
  structure.lat_column_mat = Eigen::Matrix3d::Identity();
  structure.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure.atom_info.resize(4);
  structure.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  Eigen::MatrixXd disp = Eigen::MatrixXd::Zero(3, 4);
  disp.col(0) << 0.1, 0.0, 0.0;
  disp.col(1) << 0.0, 0.1, 0.0;
  disp.col(2) << 0.0, 0.0, 0.1;
  disp.col(3) << 0.1, 0.2, 0.3;
  structure.atom_info.properties.emplace("disp", disp);

  Eigen::VectorXd Hstrain(6);
  Hstrain << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;
  structure.properties.emplace("Hstrain", Hstrain);

  Eigen::Matrix3d R(3, 3);
  R.col(0) << 1.0, 0.0, 0.0;
  R.col(1) << 0.0, 0.0, 1.0;
  R.col(2) << 0.0, 1.0, 0.0;
  Eigen::Vector3d tau(3);
  tau << 1.0, 1.0, 1.0;
  xtal::SymOp op(R, tau, false);

  apply(op, structure);

  Eigen::Matrix3d expected_lat_column_mat;
  expected_lat_column_mat.col(0) << 1.0, 0.0, 0.0;
  expected_lat_column_mat.col(1) << 0.0, 0.0, 1.0;
  expected_lat_column_mat.col(2) << 0.0, 2.0, 0.0;
  EXPECT_TRUE(
      CASM::almost_equal(structure.lat_column_mat, expected_lat_column_mat));

  Eigen::MatrixXd expected_coords = Eigen::MatrixXd::Zero(3, 4);
  expected_coords.col(0) << 0.0, 1.0, 0.0;
  expected_coords.col(1) << 0.5, 1.5, 0.5;
  expected_coords.col(2) << 0.0, 0.0, 0.0;
  expected_coords.col(3) << 0.5, 0.5, 0.5;
  EXPECT_TRUE(CASM::almost_equal(structure.atom_info.coords, expected_coords));

  Eigen::MatrixXd expected_disp = Eigen::MatrixXd::Zero(3, 4);
  expected_disp.col(0) << 0.1, 0.0, 0.0;
  expected_disp.col(1) << 0.0, 0.0, 0.1;
  expected_disp.col(2) << 0.0, 0.1, 0.0;
  expected_disp.col(3) << 0.1, 0.3, 0.2;
  EXPECT_TRUE(CASM::almost_equal(structure.atom_info.properties["disp"],
                                 expected_disp));

  Eigen::MatrixXd expected_Hstrain(6, 1);
  expected_Hstrain.col(0) << 0.01, 0.03, 0.02, 0.04, 0.06, 0.05;
  EXPECT_TRUE(
      CASM::almost_equal(structure.properties["Hstrain"], expected_Hstrain));
}

TEST(SimpleStructureIsEquivalent, Test1) {
  xtal::SimpleStructure structure1;
  structure1.lat_column_mat = Eigen::Matrix3d::Identity();
  structure1.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure1.atom_info.resize(4);
  structure1.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure1.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure1.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure1.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure1.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  // equivalent, but re-order atoms
  xtal::SimpleStructure structure2;
  structure2.lat_column_mat = Eigen::Matrix3d::Identity();
  structure2.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure2.atom_info.resize(4);
  structure2.atom_info.coords.col(1) << 0.0, 0.0, 0.0;
  structure2.atom_info.coords.col(2) << 0.5, 0.5, 0.5;
  structure2.atom_info.coords.col(3) << 0.0, 0.0, 1.0;
  structure2.atom_info.coords.col(0) << 0.5, 0.5, 1.5;
  structure2.atom_info.names = std::vector<std::string>({"B", "A", "A", "B"});

  EXPECT_TRUE(is_equivalent(structure1, structure2));
}

TEST(SimpleStructureIsEquivalent, Test2) {
  xtal::SimpleStructure structure1;
  structure1.lat_column_mat = Eigen::Matrix3d::Identity();
  structure1.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure1.atom_info.resize(4);
  structure1.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure1.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure1.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure1.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure1.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  // equivalent, but re-order and translate atoms
  xtal::SimpleStructure structure2;
  structure2.lat_column_mat = Eigen::Matrix3d::Identity();
  structure2.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure2.atom_info.resize(4);
  structure2.atom_info.coords.col(1) << 1.0, 0.0, 0.0;
  structure2.atom_info.coords.col(2) << 1.5, 0.5, 0.5;
  structure2.atom_info.coords.col(3) << 1.0, 0.0, 1.0;
  structure2.atom_info.coords.col(0) << 1.5, 0.5, 1.5;
  structure2.atom_info.names = std::vector<std::string>({"B", "A", "A", "B"});

  EXPECT_TRUE(is_equivalent(structure1, structure2));
}

TEST(SimpleStructureIsEquivalent, Test3) {
  xtal::SimpleStructure structure1;
  structure1.lat_column_mat = Eigen::Matrix3d::Identity();
  structure1.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure1.atom_info.resize(4);
  structure1.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure1.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure1.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure1.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure1.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  // not equivalent, sub-lattice vector translation
  xtal::SimpleStructure structure2;
  structure2.lat_column_mat = Eigen::Matrix3d::Identity();
  structure2.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure2.atom_info.resize(4);
  structure2.atom_info.coords.col(1) << 0.2, 0.0, 0.0;
  structure2.atom_info.coords.col(2) << 0.7, 0.5, 0.5;
  structure2.atom_info.coords.col(3) << 0.2, 0.0, 1.0;
  structure2.atom_info.coords.col(0) << 0.7, 0.5, 1.5;
  structure2.atom_info.names = std::vector<std::string>({"B", "A", "A", "B"});

  EXPECT_FALSE(is_equivalent(structure1, structure2));
}

TEST(SimpleStructureIsEquivalent, GlobalPropertiesTest1) {
  xtal::SimpleStructure structure1;
  structure1.lat_column_mat = Eigen::Matrix3d::Identity();
  structure1.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure1.atom_info.resize(4);
  structure1.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure1.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure1.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure1.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure1.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  // equivalent, but re-order atoms
  xtal::SimpleStructure structure2;
  structure2.lat_column_mat = Eigen::Matrix3d::Identity();
  structure2.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure2.atom_info.resize(4);
  structure2.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure2.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure2.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure2.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure2.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  Eigen::VectorXd Hstrain(6);
  Hstrain << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;
  structure1.properties.emplace("Hstrain", Hstrain);

  Hstrain << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;
  structure2.properties.emplace("Hstrain", Hstrain);

  EXPECT_TRUE(is_equivalent(structure1, structure2));
}

TEST(SimpleStructureIsEquivalent, LocalPropertiesTest1) {
  xtal::SimpleStructure structure1;
  structure1.lat_column_mat = Eigen::Matrix3d::Identity();
  structure1.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure1.atom_info.resize(4);
  structure1.atom_info.coords.col(0) << 0.0, 0.0, 0.0;
  structure1.atom_info.coords.col(1) << 0.5, 0.5, 0.5;
  structure1.atom_info.coords.col(2) << 0.0, 0.0, 1.0;
  structure1.atom_info.coords.col(3) << 0.5, 0.5, 1.5;
  structure1.atom_info.names = std::vector<std::string>({"A", "A", "B", "B"});

  Eigen::MatrixXd disp = Eigen::MatrixXd::Zero(3, 4);
  disp.col(0) << 0.1, 0.0, 0.0;
  disp.col(1) << 0.0, 0.1, 0.0;
  disp.col(2) << 0.0, 0.0, 0.1;
  disp.col(3) << 0.1, 0.2, 0.3;
  structure1.atom_info.properties.emplace("disp", disp);

  // equivalent, but re-order atoms
  xtal::SimpleStructure structure2;
  structure2.lat_column_mat = Eigen::Matrix3d::Identity();
  structure2.lat_column_mat.col(2) << 0.0, 0.0, 2.0;
  structure2.atom_info.resize(4);
  structure2.atom_info.coords.col(1) << 0.0, 0.0, 0.0;
  structure2.atom_info.coords.col(2) << 0.5, 0.5, 0.5;
  structure2.atom_info.coords.col(3) << 0.0, 0.0, 1.0;
  structure2.atom_info.coords.col(0) << 0.5, 0.5, 1.5;
  structure2.atom_info.names = std::vector<std::string>({"B", "A", "A", "B"});

  disp.col(1) << 0.1, 0.0, 0.0;
  disp.col(2) << 0.0, 0.1, 0.0;
  disp.col(3) << 0.0, 0.0, 0.1;
  disp.col(0) << 0.1, 0.2, 0.3;
  structure2.atom_info.properties.emplace("disp", disp);

  EXPECT_TRUE(is_equivalent(structure1, structure2));
}
