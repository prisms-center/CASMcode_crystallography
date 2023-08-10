#include "testdir.hh"

#include "autotools.hh"
#include "gtest/gtest.h"

namespace test {

/// Store data files in "tests/unit/<module_name>/data/<file_name>"
fs::path data_dir(std::string module_name) {
  // abs_srcdir == "<project path>/tests"
  return fs::path{autotools::abs_srcdir()} / "unit" / module_name / "data";
}

/// Store data files in "tests/unit/<module_name>/<file_name>"
fs::path data_file(std::string module_name, std::string file_name) {
  return data_dir(module_name) / file_name;
}

TmpDir::TmpDir() : m_remove_on_destruction(true) {
  fs::path init = fs::current_path() / "CASM_test_projects" / "tmp";
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

}  // namespace test
