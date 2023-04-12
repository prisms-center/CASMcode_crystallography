Installation
============

Currently, libcasm-xtal must be built from source, first installing the required CASM C++ libraries, then building and installing the Python package.

Installation of libcasm-xtal requires:

- git, GNU autotools for building C++ libraries
- Python >=3.8, gitpython
- The compatible version of the CASM C++ global and crystallography libraries must be installed.
- Development environment that allows compiling the pybind11 interface to CASM C++ (i.e. C++ compiler with support for c++17)


Compiling CASM C++ libraries
----------------------------

An example is given for CASMcode_global. Repeat as necessary to install additional CASM libraries.

Clone `CASMcode_global <https://github.com/prisms-center/CASMcode_global>`_ and its submodules:

::

    git clone https://github.com/prisms-center/CASMcode_global.git
    cd CASMcode_global
    git submodule init
    git submodule update
    bash ./bootstrap.sh
    mkdir build && cd build

Create a configuration script, ``build/local-configure.sh``. Depending on your system, some customization my be required. Here is an example for building in a conda environment with ``ccache`` installed:

::

    CASM_PREFIX=$CONDA_PREFIX   # Where to install (/usr/local, $HOME/.local, etc.)

    CASM_CXXFLAGS="-O3 -Wall -DNDEBUG -I${CASM_PREFIX}/include"
    CASM_CC="ccache cc"
    CASM_CXX="ccache c++"
    CASM_PYTHON="python"
    CASM_LDFLAGS="-L$CASM_PREFIX/lib"
    CASM_CONFIGFLAGS="--prefix=$CASM_PREFIX "

    ../configure CXXFLAGS="${CASM_CXXFLAGS}" CC="${CASM_CC}" CXX="${CASM_CXX}" PYTHON="${CASM_PYTHON}" LDFLAGS="${CASM_LDFLAGS}" ${CASM_CONFIGFLAGS}

Configure, make, run tests, install:

::

    bash ./local-configure.sh
    make -j4  # parallel build with 4 processors
    make check -j4
    make install


Install the libcasm-xtal Python package
---------------------------------------

Go to the ``python`` directory:

::

    cd ../python

Normal installation:

::

    pip install .

Editable installation:

::

    pip install -e .
