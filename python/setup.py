import glob
import os
import sys

__version__ = "2.0.dev1"

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

with open(os.path.join('README.md'), encoding='utf-8') as f:
    long_description = f.read()

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

ext_modules = [
    Pybind11Extension("casm.xtal",
        ["src/xtal.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        cxx_std=17,
        library_dirs = [
            '/Users/bpuchala/.local/conda/envs/casm_modules_2.X/lib'
        ],
        include_dirs = [
            '/Users/bpuchala/.local/conda/envs/casm_modules_2.X/include/casm/external',
            '/Users/bpuchala/.local/conda/envs/casm_modules_2.X/include'
        ],
        extra_compile_args=['-D_LIBCPP_DISABLE_AVAILABILITY', '--std=c++17'],
        extra_link_args=['-lcasm_global','-lcasm_crystallography']
        ),

]


setup(
    name='casm-xtal',
    version=__version__,
    url='https://github.com/prisms-center/CASMcode_mapping',
    description='CASM structure mapping Python interface',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='CASM developers',
    author_email='casm-developers@lists.engr.ucsb.edu',
    license='LGPL2.1+',
    #packages=['casm.xtal'],
    install_requires=[
        "pybind11", "numpy"
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    data_files=[('', ['LICENSE'])],
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",)
