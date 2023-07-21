## Contributing

Collaboration is welcome and new features can be incorporated by forking the repository on GitHub, creating a new feature, and submitting pull requests. If you are interested in developing features that involve a significant time investment we encourage you to first contact the CASM development team at <casm-developers@lists.engr.ucsb.edu>.

Pull requests should:

- Create a branch from either the development branch for new features (i.e. `2.X` -> `2.X-myfeature`) or the appropriate patch branch for bug fixes `2.0-patch-issue`)
- Propose a minimal set of changes
- Have code formatted and documented as described below
- Include appropriate tests
- Pass all CI tests
- Include a suggested CHANGELOG.md entry, see [keepachangelog.com](https://keepachangelog.com).

## Installing from source

> **Note**
> Care must be taken, especially on linux, that code linking to the CASM C++ libraries is compiled with the same choice of the -D_GLIBCXX_USE_CXX11_ABI compiler flag, otherwise there will be "undefined reference" linking errors (see [Dual ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)).  The CASM Linux C++ libraries distributed on PyPI with tag "manylinux2014" use -D_GLIBCXX_USE_CXX11_ABI=0. Newer compilers, and all supported macosx versions, default to -D_GLIBCXX_USE_CXX11_ABI=1.  It will be noted in the following where configuration steps may be necessary.


Installation of `libcasm-xtal` from source requires standard compilers with support for C++17, Python >= 3.8. For example:

- On Ubuntu linux:

  ```
  sudo apt-get install build-essential cmake
  ```

- On Mac OSX:

  ```
  xcode-select --install
  brew install cmake
  ```

- In a conda environment:

  ```
  conda create -n casm --override-channels -c conda-forge python=3 cmake
  conda activate casm
  ```

Then `libcasm-xtal` and its dependencies can be installed with:

    # Configuration options:
    #
    # To install with manylinux2014 package dependencies set:
    #   export SKBUILD_CONFIGURE_OPTIONS="-DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'"
    # To print compiler commands set:
    #   export SKBUILD_BUILD_OPTIONS="--verbose"
    pip install .


## Building documentation

Install documentation requirements:

    pip install -r doc_requirements.txt

Install `libcasm-xtal` first, then build and open the documentation:

    cd python/doc
    sphinx-build -b html . _build/html
    open _build/html/index.html


## Testing

To install testing requirements, do:

    pip install -r test_requirements.txt

Use `pytest` to run the tests. To run all tests, do:

    pytest -rsap python/tests

As an example of running a specific test, do:

    pytest -rsap python/tests/<filepath>::<function_name>


### Overview

Building is performed using [scikit-build](https://scikit-build.readthedocs.io/en/latest/index.html) to create distributions that include both the C++ and Python portions of CASM, which are linked using [pybind11](https://pybind11.readthedocs.io).

The project is organized as follows:

- `include/`: C++ headers
- `src/`: C++ source files
- `tests/unit`: C++ unit tests
- `python/libcasm/<name>/`: Python namespace packages
- `python/src/`: pybind11 wrappers
- `python/tests/<name>/`: Python tests
- `python/doc/`: Python documentation

When the project is built and installed, components are added to the Python installation location (i.e. `<python package prefix> = <something>/lib/pythonX.Y/sites-packages/`) in the `libcasm/` folder at the following locations:

`<python package prefix>/libcasm/`:

- `include/`: C++ headers
- `lib/`: Built C++ libraries (i.e. `libcasm_crystallography.so` or `libcasm_crystallography.dylib`)
- `share/CASMcode_<name>/cmake/`: CMake distribution data
- `<packagename>/`: CASM Python namespace packages, with Python source files and built pybind11 wrapper libraries

#### Adding or removing files

CMake is used to build the project. When files are added or removed from the project, the `CMakeLists.txt` and `tests/CMakeList.txt` files must be updated. This can be done in an automated fashion using a script to copy the `CMakeLists.txt.in` and `tests/CMakeList.txt.in` templates and populate them with the current project files:

    python make_CMakeLists.py

The files to be included in source distributions are specified by the `MANIFEST.in` file. This should rarely need to be changed.

## Installing in editable mode

Editable installation of the pure Python components (i.e. `pip install -e`) is very useful for development. It is not currently supported by scikit-build or [scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/) (the next generation of scikit-build) using the standard build process, but may be possible soon.

Currently, editable installation can be done by first installing the pure C++ portion of a CASM module and then installing the pybind11 wrapper and Python portions of the module.

#### Installing C++ components only

Install build dependencies:

    pip install -r build_requirements.txt

Once the `libcasm-global` Python package is installed by the previous step, CMake should be able find and link to other CASM module dependencies in the `<python package prefix>/libcasm/` directory automatically. It will also set that as the install location unless the user defines `CMAKE_INSTALL_PREFIX` explicitly to override it. CMake will also detect and use `ccache` if it is installed, which is very useful to avoid recompiling objects and greatly speed up development.

Then, to make and install CASM C++ components in the `<python package prefix>/libcasm/` directory:

    mkdir build_cxx_only
    cd build_cxx_only

    # Some configuration options:
    #
    # To use an existing C++ only casm installation add:
    #   -DCASM_PREFIX=<path-to-casm>
    # To link to manylinux2014 packages add:
    #   -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'
    # CASM may be slow if not using the Release build type (-O3 -DNDEBUG).
    # To specify build type use Release, Debug, RelWithDebInfo, or MinSizeRel
    #   -DCMAKE_BUILD_TYPE=Release
    #
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j4 VERBOSE=1
    make install

C++ unit tests can be built after C++ components are installed as in the previous step. To make and run unit tests, return to the repository root directory and do:

    mkdir build_cxx_test
    cd build_cxx_test
    # Some configuration options:
    #
    # To use an existing C++ only casm installation add:
    #   -DCASM_PREFIX=<path-to-casm>
    # To link to manylinux2014 packages add:
    #   -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'
    # CASM tests may be slow if not using the Release build type (-O3 -DNDEBUG).
    # To specify build type use Release, Debug, RelWithDebInfo, or MinSizeRel
    #   -DCMAKE_BUILD_TYPE=Release
    #
    cmake -DCMAKE_BUILD_TYPE=Release ../tests
    make -j4 VERBOSE=1
    make test

To uninstall a C++ only installation, use the `install_manifest.txt` file generated by `cmake`:

    cd build_cxx_only
    xargs rm < install_manifest.txt

#### Installing pybind11 and pure Python components

Use the `setup.py` file in `CASMcode_crystallography/python/` for editable install of the pure Python components:

    cd python
    # It is required to set the casm prefix:
    export CASM_PREFIX=$(python -m libcasm.casmglobal --prefix)

    # Some configuration options:
    #
    # To link to manylinux2014 packages set:
    #   export CASM_EXTRA_COMPILE_ARGS='-D_GLIBCXX_USE_CXX11_ABI=0'
    #
    pip install -e .

At this point, changes made to pure Python source files are immediately testable. Testing changes made to the pybind11 wrappers requires re-building with `pip install -e .`. Testing changes made to the CASM C++ components requires re-building and re-installing the C++ components and then re-building with `pip install -e .`.

To uninstall the Python package do:

    pip uninstall libcasm-xtal


## Testing the combined build process

To test the combined build process performed by `pip install .`, it may be useful to skip build isolation:

    pip install -v --no-build-isolation .

This allows `ccache` to identify and reuse identical compilation steps and speeds up the build process. To do this, it is necessary to have installed all build requirements already from `build_requirements.txt`. Build isolation should not be skipped for CI tests.

When built together using pip, all C++ and Python components can be uninstalled using:

    pip uninstall libcasm-xtal


## Formatting and style

To install formatting requirements, do:

    pip install -r dev_requirements.txt


#### Python formatting

For Python code formatting, use black. Do:

    black python/

#### Python docstring style

- When in doubt, refer to [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html), [pandas](https://pandas.pydata.org/docs/development/contributing_documentation.html), or [scikit-learn](https://scikit-learn.org/dev/developers/contributing.html#documentation).
- When referring to constructor arguments or function variables in docstring text, use the convention ``` `variable` ```, so variables appear italicized because (i.e. The *variable* is important).
- When describing that a variable has a particular value or how it is used in a code snippet, then use either inline code (```variable=True```) or a code block:

  ```
  .. code-block:: Python

      variable = 6
  ```
- Make use of ```.. rubric:: Special Methods``` to create a section in a class docstring to document any special members of a class, such as comparison operators (`<`, `<=`, `>`, `>=`, etc.) or overloaded operators (`*`, `+`, `+=`, `-`, `-=`, etc.).


#### C++ formatting

For C++ code formatting, use clang-format with `-style=google`. Use the `stylize.sh` script to format files staged for commit. To enforce that it is run before committing, place the `pre-commit` file, or equivalent code, in `.git/hooks`. The process looks like:

    git add <unformatted new and changed files>
    ./stylize.sh
    git add <formatted new and changed files>

    # good idea to check changes
    git status
    git diff --cached
    git commit -m "Added new feature X"

If C++ files have been added or removed from `include/`, `src/`, or `tests/`, then `CMakeLists.txt` should be updated from the `CMakeLists.txt.in` template using:

    python make_CMakeLists.py


#### C++ style

- When in doubt, refer to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use doxygen for comments.
- Fit in to the existing style for a particular file


## Adding tests

Python:
- Add Python tests for `libcasm.<subpackage>` in `python/tests/<subpackage>`, using pytest.
- If data files are needed for testing, they can be placed in `python/tests/<subpackage>/data/`.
- To access data files use the `shared_datadir` fixture available from the [`pytest-datadir`](https://pypi.org/project/pytest-datadir/) plugin.
- To create temporary testing directories for reading and writing files, use the [`tmpdir` and `tmpdir_factory`](https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#the-tmpdir-and-tmpdir-factory-fixtures) fixtures available from pytest.
- For tests that involve an expensive setup process, such as compiling Clexulators, a session-length shared datadir can be constructed once and re-used as done [here](https://github.com/prisms-center/CASMcode_clexulator/blob/2.X/python/tests/clexulator/conftest.py) in CASMcode_clexulator.
- Expensive tests can also be set to run optionally using flags as demonstrated in CASMcode_clexulator.

C++:
- Add C++ library tests in `tests/unit/<module>`, with the naming convention `<something>_test.cpp`, using googletest.
- If data files are needed for testing, they can be placed in `tests/unit/<module>/data/`.
- To access data files and create temporary testing directories for reading and writing files, use the methods available in `tests/unit/testdir.hh`.
- If a new module is added, (i.e. a new `casm/<module>`) then new unit tests should be added under `tests/unit/<module/` and `tests/CMakeLists.txt.in` and `make_CMakeLists.py` must be updated to add the new unit tests.
- To run only C++ library tests, follow the example for building only the C++ library.


### Using CASM C++ libraries

A `__main__.py` file added to the `libcasm.casmglobal` package allows finding the CASM installation location for use by other packages.

To find the installation location (i.e. `<python package prefix>/libcasm`) use:

    python -m libcasm.casmglobal --prefix

As an example using this with CMake to configure and build a program using CASM C++ libraries see:
- [tests/CMakeLists.txt.in](tests/CMakeLists.txt.in): The unit tests provide an example of configuring a program to link with CASM C++ libraries `libcasm_global` and `libcasm_crystallography`.
- [CMakeLists.txt.in](CMakeLists.txt.in): The main configuration file provides an example of configuring a C++ library to link with `libcasm_global` and update the default installation prefix to be in the same location as other CASM tools. It also demonstrates how to configure a Python extension module built with pybind11 to link with CASM C++ libraries.
