cmake_minimum_required(VERSION 3.20)

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# Add project_options from https://github.com/aminya/project_options
# Change the version in the following URL to update the package (watch the releases of the repository for future updates)
set(PROJECT_OPTIONS_VERSION "v0.35.1")
FetchContent_Declare(
  _project_options
  URL https://github.com/aminya/project_options/archive/refs/tags/${PROJECT_OPTIONS_VERSION}.zip)
FetchContent_MakeAvailable(_project_options)
include(${_project_options_SOURCE_DIR}/Index.cmake)


# Set version
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" SCOPI_VERSION)
    string(STRIP "${SCOPI_VERSION}" SCOPI_VERSION)
else()
message(FATAL_ERROR "File ${CMAKE_CURRENT_SOURCE_DIR}/version.txt not found")
endif()

project(scopi VERSION ${SCOPI_VERSION} LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

project_options(
    PREFIX "scopi"
    ENABLE_VS_ANALYSIS
)

# Options
# =======
OPTION(SCOPI_USE_TBB "enable TBB" OFF)
OPTION(SCOPI_USE_OPENMP "enable OpenMP" OFF)
OPTION(BUILD_EXAMPLES "scopi examples" OFF)
OPTION(BUILD_TESTS "scopi test suite" OFF)

if(SCOPI_USE_TBB AND SCOPI_USE_OPENMP)
    message(
        FATAL
        "SCOPI_USE_TBB and SCOPI_USE_OPENMP cannot both be active at once"
        )
    endif()

# Dependencies
# ============
# set(DEPENDENCIES_CONFIGURED xtensor xtensor-blas nlohmann_json fmt CLI11 plog nanoflann)

# if(SCOPI_USE_TBB)
#     set(XTENSOR_USE_TBB 1)
#     append(DEPENDENCIES_CONFIGURED TBB)
# endif()

# if(SCOPI_USE_OPENMP)
#     append(DEPENDENCIES_CONFIGURED OpenMP)
# endif()

# foreach(DEPENDENCY ${DEPENDENCIES_CONFIGURED})
#     find_package(${DEPENDENCY} REQUIRED)
# endforeach()

set(SCOPI_SRC
    src/vap/vap_fpd.cpp
    src/vap/vap_projection.cpp
    src/minpack.cpp
    src/params.cpp
    src/quaternion.cpp
    src/utils.cpp
)

add_library(libscopi ${SCOPI_SRC})

# link project_options/warnings
target_link_libraries(libscopi
  PRIVATE scopi_project_options scopi_project_warnings
)

# Includes:
target_include_interface_directories(libscopi "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Find dependencies:
target_find_dependencies(libscopi
  PRIVATE_CONFIG
  xtensor
  xtensor-blas
  nlohmann_json
  fmt
  CLI11
  plog
  nanoflann
)

# Link dependencies:
target_link_system_libraries(libscopi
  PRIVATE
  xtensor
  xtensor-blas
  nlohmann_json::nlohmann_json
  fmt::fmt
  CLI11::CLI11
  plog::plog
  nanoflann::nanoflann
)

# target_link_libraries(scopi PUBLIC
#     xtensor
#     xtensor-blas
#     nlohmann_json::nlohmann_json
#     fmt::fmt
#     CLI11::CLI11
#     plog::plog
#     nanoflann::nanoflann
# )

# if(SCOPI_USE_TBB)
#     target_link_libraries(scopi PUBLIC TBB::tbb)
#     target_compile_definitions(scopi PUBLIC SCOPI_USE_TBB)
# endif()

# if(SCOPI_USE_OPENMP)
#     target_link_libraries(scopi PUBLIC OpenMP::OpenMP_CXX)
#     target_compile_definitions(scopi PUBLIC SCOPI_USE_OPENMP)
# endif()

if(BUILD_EXAMPLES)
    add_subdirectory(demos)
else()
    add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

# Package the project
package_project(
  TARGETS libscopi
  scopi_project_options scopi_project_warnings
)
