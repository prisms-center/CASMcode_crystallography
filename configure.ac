#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([CASM],
        m4_esyscmd([tr -d '\n' < ./casm_version.txt]),
        [casm-developers@lists.engr.ucsb.edu],
        [],
        [https://github.com/prisms-center/CASMcode_crystallography])
AC_CONFIG_AUX_DIR([build-aux]) #This is my doing
AC_CONFIG_MACRO_DIR([build-aux/m4])
#Rules to apply to every Makefile.am. Using tar-ustar due to loooooong file names (I'm looking at you Eigen)
AM_INIT_AUTOMAKE([foreign subdir-objects 1.9 tar-ustar])


AC_CONFIG_SRCDIR([src/casm/crystallography])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP

## Checks for libraries.
AC_SEARCH_LIBS([dlopen], [dl], [], AC_MSG_ERROR(dlopen from dl library not found!))
AX_CHECK_ZLIB(,[AC_MSG_ERROR([Could not find zlib])])

#I added this
LT_INIT([shared disable-static])
AC_PROG_LIBTOOL
LT_LIB_DLLOAD
AC_DISABLE_STATIC

# AX_CXX_COMPILE_STDCXX_11
AX_CXX_COMPILE_STDCXX_17

#Require at least this version
AM_PATH_PYTHON([3.6])

# Checks for header files.
AC_CHECK_HEADERS([float.h limits.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_C_RESTRICT
AC_TYPE_SIZE_T
AC_TYPE_UINT64_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
AC_FUNC_ERROR_AT_LINE
AC_FUNC_MALLOC
AC_HEADER_MAJOR
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([floor iswprint memmove memset mkdir pow rmdir select sqrt strchr strpbrk strrchr strstr strtol])

#Generate all the necessary files from *.in
AC_CONFIG_FILES([Makefile])

AC_OUTPUT
