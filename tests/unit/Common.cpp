#include "Common.hh"

#include <chrono>
#include <thread>

#include "autotools.hh"
#include "casm/crystallography/BasicStructure.hh"
#include "casm/crystallography/Site.hh"
#include "gtest/gtest.h"

namespace test {

/// \brief Check expected JSON vs calculated JSON using BOOST_CHECK_EQUAL
///
/// Checks:
/// \code
/// if(expected.contains(test)) {
///   BOOST_CHECK(expected[test].almost_equal, calculated);
/// }
/// \endcode
///
/// If \code !expected.contains(test) && !quiet \endcode, print the calculated
/// JSON so that it can be added to the test data.
void check(std::string test, const jsonParser &expected,
           const jsonParser &calculated, fs::path test_cases_path, bool quiet,
           double tol) {
  if (!expected.contains(test) && !quiet) {
    std::cout << "Test case: " << expected["title"] << " has no \"" << test
              << "\" test data." << std::endl;
    std::cout << "To use the current CASM results, add the following to the "
              << expected["title"] << " test case in " << test_cases_path
              << std::endl;
    jsonParser j = jsonParser::object();
    j[test] = calculated;
    std::cout << j << std::endl;
  }

  bool ok;
  fs::path diff_path;
  if (tol == 0.0) {
    ok = (expected[test] == calculated);
    if (!ok) {
      diff_path = find_diff(expected[test], calculated);
    }
  } else {
    ok = expected[test].almost_equal(calculated, tol);
    if (!ok) {
      fs::path diff_path = find_diff(expected[test], calculated, tol);
    }
  }

  if (!ok) {
    std::cout << "Difference at: " << diff_path << std::endl;
    std::cout << "Expected: \n"
              << expected[test].at(diff_path) << "\n"
              << "Found: \n"
              << calculated.at(diff_path) << std::endl;
  }

  EXPECT_EQ(ok, true);

  return;
}

/// Store data files in "tests/unit/<module_name>/data/<file_name>"
fs::path data_dir(std::string module_name) {
  return fs::path{autotools::abs_srcdir()} / "tests" / "unit" / module_name /
         "data";
}

/// Store data files in "tests/unit/<module_name>/<file_name>"
fs::path data_file(std::string module_name, std::string file_name) {
  return data_dir(module_name) / file_name;
}

TmpDir::TmpDir() : m_remove_on_destruction(true) {
  fs::path init =
      fs::path{autotools::abs_srcdir()} / "tests" / "test_projects" / "tmp";
  fs::path result = init;
  int index = 0;
  std::string dot = ".";
  while (!fs::create_directories(result)) {
    result = fs::path(init.string() + dot + std::to_string(index));
    ++index;
  }

  m_path = result;
}

/// Sets flag to remove the temp directory on destruction (set by default)
void TmpDir::remove_on_destruction() { m_remove_on_destruction = true; }

/// Unsets flag to remove the temp directory on destruction
void TmpDir::do_not_remove_on_destruction() { m_remove_on_destruction = false; }

TmpDir::~TmpDir() {
  if (m_remove_on_destruction) {
    fs::remove_all(m_path);
  }
}

fs::path TmpDir::path() const { return m_path; }

TmpDir::operator fs::path() const { return m_path; }

/// \brief Create a new project directory, appending ".(#)" to ensure
/// it is a new project
fs::path proj_dir(fs::path init) {
  init = fs::absolute(init);
  fs::path result = init;
  int index = 0;
  std::string dot = ".";
  while (!fs::create_directories(result)) {
    result = fs::path(init.string() + dot + std::to_string(index));
    ++index;
  }

  return result;
}

/// \brief Check some aspects of a SymGroup json
void check_symgroup(const jsonParser &json, int N_op, int N_class) {
  // temporarily disabled while transiting character table determination
  // EXPECT_EQ(json["character_table"].size(), N_class);

  // EXPECT_EQ(json["conjugacy_class"].size(), N_class);

  EXPECT_EQ(json["group_operations"].size(), N_op);
  EXPECT_EQ(
      (*json["group_operations"].begin())["info"]["type"].get<std::string>(),
      "identity");

  EXPECT_EQ(json["group_structure"]["multiplication_table"].size(), N_op);
  for (auto i = 0; i < json["group_structure"]["multiplication_table"].size();
       ++i) {
    EXPECT_EQ(json["group_structure"]["multiplication_table"][i].size(), N_op);
  }
}

}  // namespace test
