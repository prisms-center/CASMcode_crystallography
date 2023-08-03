import copy
import os


def is_extensionless_Eigen_header(filepath):
    """Returns true if the provided file resides in
    include/casm/external/Eigen, which contains header files
    that don't have an extension like .h

    Parameters
    ----------
    filepath : path to file, relative to git root

    Returns
    -------
    bool

    """
    parent, f = os.path.split(filepath)
    return parent == "include/casm/external/Eigen"


def header_extensions():
    """List of extensions that are considered header files:
    .h, .hh, .hpp
    Returns
    -------
    list of str

    """
    return [".h", ".hh", ".hpp"]


def source_extensions():
    """List of extensions that are considered source files:
    .c, .cc, .cxx, .cpp
    Returns
    -------
    list of str

    """
    return [".c", ".cc", ".cxx", ".cpp", ".C"]


def has_header_extension(filepath):
    """Returns true of the file has an extension such as
    .h, .hh, etc as defined by header_extensions()

    Parameters
    ----------
    filepath : path to file

    Returns
    -------
    bool

    """
    for ext in header_extensions():
        if filepath.endswith(ext):
            return True
    return False


def has_source_extension(filepath):
    """Returns true of the file has an extension such as
    .c, .cc, etc as defined by source_extensions()

    Parameters
    ----------
    filepath : path to file

    Returns
    -------
    bool

    """
    for ext in source_extensions():
        if filepath.endswith(ext):
            return True
    return False


def header_and_source_extensions():
    return header_extensions() + source_extensions()


def header_files(search_root):
    files_by_dir = [
        (dirpath, files)
        for dirpath, dirnames, files in os.walk(search_root, followlinks=True)
    ]
    files = [
        os.path.join(dirpath, file) for dirpath, files in files_by_dir for file in files
    ]
    _header_files = [
        file
        for file in files
        if is_extensionless_Eigen_header(file) or has_header_extension(file)
    ]
    return _header_files


def source_files(search_root):
    files = [
        (dirpath, files)
        for dirpath, dirnames, files in os.walk(search_root, followlinks=True)
    ]
    _source_files = [
        os.path.join(d, f) for d, fs in files for f in fs if has_source_extension(f)
    ]
    return _source_files


def libcasm_testing_source_files(search_dir):
    files = [
        os.path.join(search_dir, file)
        for file in os.listdir(search_dir)
        if file != "gtest_main_run_all.cpp" and has_source_extension(file)
    ]
    return files


def unit_test_source_files(search_dir, additional):
    files = copy.copy(additional)
    files += [
        os.path.join(search_dir, file)
        for file in os.listdir(search_dir)
        if has_source_extension(file)
    ]
    return files


def as_cmake_file_strings(files):
    cmake_file_strings = ""
    for file in files:
        cmake_file_strings += "  ${PROJECT_SOURCE_DIR}/" + str(file) + "\n"
    return cmake_file_strings


### make CMakeLists.txt from CMakeLists.txt.in ###

with open("CMakeLists.txt.in", "r") as f:
    cmakelists = f.read()

files = header_files("include")
cmake_file_strings = as_cmake_file_strings(files)
cmakelists = cmakelists.replace("@header_files@", cmake_file_strings)

files = source_files("src")
cmake_file_strings = as_cmake_file_strings(files)
cmakelists = cmakelists.replace("@source_files@", cmake_file_strings)

with open("CMakeLists.txt", "w") as f:
    f.write(cmakelists)


### make tests/CMakeLists.txt from tests/CMakeLists.txt.in ###

os.chdir("tests")
with open("CMakeLists.txt.in", "r") as f:
    cmakelists = f.read()

files = libcasm_testing_source_files("unit")
cmake_file_strings = as_cmake_file_strings(files)
cmakelists = cmakelists.replace("@libcasm_testing_source_files@", cmake_file_strings)


additional = ["unit/gtest_main_run_all.cpp"]
files = unit_test_source_files("unit/crystallography", additional)
cmake_file_strings = as_cmake_file_strings(files)
cmakelists = cmakelists.replace(
    "@casm_unit_crystallography_source_files@", cmake_file_strings
)

with open("CMakeLists.txt", "w") as f:
    f.write(cmakelists)
