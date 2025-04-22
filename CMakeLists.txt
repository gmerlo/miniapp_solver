cmake_minimum_required(VERSION 3.15)

project("PETSc Solver Miniapp"
  LANGUAGES C CXX
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MINIAPP_DATA_PATH $ENV{MINIAPP_DATA_PATH})
if(MINIAPP_DATA_PATH)
  message(STATUS "Data path set to ${MINIAPP_DATA_PATH}")
else()
  message(WARNING "Data path not set. ")
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/src/config.hpp.in
    ${CMAKE_BINARY_DIR}/include/config.hpp
    @ONLY
)

find_package(MPI REQUIRED)
find_package(PETSc REQUIRED)
find_package(Ginkgo REQUIRED)
find_package(HDF5 REQUIRED)

include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

add_executable(petsc_solver_miniapp
  src/petsc_solver_miniapp.cpp
)

target_include_directories(petsc_solver_miniapp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(petsc_solver_miniapp PRIVATE
  MPI::MPI_CXX
  petsc::petsc
  yaml-cpp::yaml-cpp
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# The nlohmann_json library is needed for the Ginkgo file config.
# If it is not available, it is fetched from github.
find_package(nlohmann_json 3.9.1 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "Fetching external nlohmann_json")
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.9.1
  )
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  set(JSON_Install OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_executable(gko_solver_miniapp
  src/gko_solver_miniapp.cpp
)

target_include_directories(gko_solver_miniapp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(gko_solver_miniapp PRIVATE
  MPI::MPI_CXX
  petsc::petsc
  Ginkgo::ginkgo
  yaml-cpp::yaml-cpp
  nlohmann_json::nlohmann_json
)

file(GLOB config_list RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" gko-config/*.json)
foreach(config IN LISTS config_list)
    configure_file("${config}" "${config}" COPYONLY)
endforeach()

#add_executable(petsc_binary_to_hdf5_converter
#  src/petsc_binary_to_hdf5_converter.cpp
#)

#target_include_directories(petsc_binary_to_hdf5_converter PRIVATE
#  ${CMAKE_CURRENT_SOURCE_DIR}/src
#)

#target_link_libraries(petsc_binary_to_hdf5_converter PRIVATE
#  MPI::MPI_CXX
#  petsc::petsc
#  HDF5::HDF5
#)
