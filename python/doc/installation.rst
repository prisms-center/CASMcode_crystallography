Installation
============

casm-xtal is currently in development as a standalone library and requires user compilation.

Installation of casm-xtal requires:

- git, GNU autotools for building C++ libraries
- Python >=3.8, gitpython
- The compatible version of the CASM C++ global and crystallography libraries must be installed.
- Development environment that allows compiling the pybind11 interface to CASM C++ (i.e. C++ compiler with support for c++17)


Compiling CASM C++ libraries
----------------------------

An example is given for CASMcode_global. Repeat as necessary to intall CASMcode_global, and CASMcode_crystallography.

Clone `CASMcode_global <https://github.com/prisms-center/CASMcode_global>`:

    git clone https://github.com/prisms-center/CASMcode_global.git
    cd CASMcode_global
    bash ./bootstrap.sh
    mkdir build && cd build

Create a configuration script in `build/local-configure.sh`. Here is an example for building on Mac in a conda environment:

    CASM_PREFIX=$CONDA_PREFIX   # Where to install (/usr/local, $HOME/.local, etc.)

    CASM_CXXFLAGS="-O3 -Wall -fPIC --std=c++17 -DNDEBUG -fcolor-diagnostics -Wno-deprecated-register -Wno-ignored-attributes -Wno-deprecated-declarations"
    CASM_CC="ccache cc"
    CASM_CXX="ccache c++"
    CASM_PYTHON="python"
    CASM_LDFLAGS="-Wl,-rpath,$CASM_PREFIX/lib"
    CASM_CONFIGFLAGS="--prefix=$CASM_PREFIX --with-zlib=$CASM_PREFIX "

    ../configure CXXFLAGS="${CASM_CXXFLAGS}" CC="$CASM_CC" CXX="$CASM_CXX" PYTHON="$CASM_PYTHON" LDFLAGS="${CASM_LDFLAGS}" ${CASM_CONFIGFLAGS}

Configure, make, run tests, install:

    bash ./local-configure.sh
    make
    make check
    make install


Install casm-xtal
-----------------

Go to `python directory`:

    cd CASMcode_crystallography/python

Normal installation:

    pip install .

Editable installation:

    pip install -e .
