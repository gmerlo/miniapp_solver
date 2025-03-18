#
# @file FindPETSc.cmake
#
# @brief Find PETSc library
#
# @copyright
# © All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
#   Swiss Plasma Center, 2021.
#
# @author
# (in alphabetical order)
# @author Emmanuel Lanti <emmanuel.lanti@epfl.ch>
# @author Nicolas Richart <nicolas.richart@epfl.ch>
#

# - Try to find PETSc
#  PETSC_FOUND         - system has PETSc
#  PETSC_INCLUDE_DIRS  - the PETSc include directories
#  PETSC_LIBRARIES     - Link these to use PETSc
#  PETSC_VERSION       - Version string (MAJOR.MINOR.SUBMINOR)

if(PETSc_FIND_REQUIRED)
  find_package(PkgConfig REQUIRED)
else()
  find_package(PkgConfig QUIET)
  if(NOT PKG_CONFIG_FOUND)
    return()
  endif()
endif()

set(_target $ENV{PE_PETSC_PKGCONFIG_LIBS})
if (_target)
  # on cray machine
  string(TOLOWER "$ENV{PE_ENV}" _pe)
  string(REGEX MATCH "(.*)_(.*)" _out "${_target}")
  set(_prefix ${CMAKE_MATCH_1})
  set(_suffix ${CMAKE_MATCH_2})
  set(_libs "${_prefix}_${_pe}_${_suffix}")
  # newest (2022) Cray environments are called crayclang instead of cray
  if (${_pe} MATCHES "cray")
    list(APPEND _libs "${_prefix}_${_pe}clang_${_suffix}")
  endif()

  find_library(PETSC_LIBRARIES  ${_libs}
      PATHS "${CRAY_PETSC_PREFIX_DIR}" ENV CRAY_PETSC_PREFIX_DIR
      PATH_SUFFIXES lib
      NO_CMAKE_PATH
      NO_DEFAULT_PATH)
  find_path(PETSC_INCLUDE_DIRS "petsc.h"
      PATHS "${CRAY_PETSC_PREFIX_DIR}" ENV CRAY_PETSC_PREFIX_DIR
      PATH_SUFFIXES include
      NO_CMAKE_PATH
      NO_DEFAULT_PATH)

  add_library(petsc::petsc INTERFACE IMPORTED)
  set_property(TARGET petsc::petsc PROPERTY INTERFACE_LINK_LIBRARIES ${PETSC_LIBRARIES})
  set_property(TARGET petsc::petsc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PETSC_INCLUDE_DIRS})
else()
  pkg_search_module(_petsc PETSc)

  # Some debug code
  # get_property(_vars DIRECTORY PROPERTY VARIABLES)
  # foreach(_var ${_vars})
  #   if (_var MATCHES "^_petsc")
  #     message("${_var} -> ${${_var}}")
  #   endif()
  # endforeach()

  if(_petsc_FOUND AND _petsc_VERSION)
    set(PETSC_VERSION ${_petsc_VERSION})
  endif()

  if(_petsc_FOUND)
    set(_petsc_libs)
    foreach(_lib ${_petsc_LIBRARIES})
      string(TOUPPER "${_lib}" _u_lib)
      find_library(PETSC_LIBRARY_${_u_lib} ${_lib} PATHS ${_petsc_LIBRARY_DIRS})
      list(APPEND _petsc_libs ${PETSC_LIBRARY_${_u_lib}})
      mark_as_advanced(PETSC_LIBRARY_${_u_lib})
    endforeach()

    if (NOT _petsc_INCLUDE_DIRS)
      pkg_get_variable(_petsc_INCLUDE_DIRS ${_petsc_MODULE_NAME} includedir)
    endif()

    set(_include_dirs)
    foreach (_path ${_petsc_INCLUDE_DIRS})
      if(EXISTS ${_path})
        list(APPEND _include_dirs ${_path})
      endif()
    endforeach()

    set(PETSC_LIBRARIES ${_petsc_libs} CACHE FILEPATH "")
    set(PETSC_INCLUDE_DIRS ${_include_dirs} CACHE PATH "")

    add_library(petsc::petsc INTERFACE IMPORTED)
    set_property(TARGET petsc::petsc PROPERTY INTERFACE_LINK_LIBRARIES ${PETSC_LIBRARIES})
    set_property(TARGET petsc::petsc PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PETSC_INCLUDE_DIRS})
  endif()
endif()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
  REQUIRED_VARS PETSC_LIBRARIES PETSC_INCLUDE_DIRS
  VERSION_VAR PETSC_VERSION)
