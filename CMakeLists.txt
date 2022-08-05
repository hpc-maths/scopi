cmake_minimum_required(VERSION 3.15)

project(scopi)

# Flags
# ===========

# set(CMAKE_SKIP_BUILD_RPATH FALSE)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic ")
set (CMAKE_CXX_STANDARD 17)
message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(SCOPI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

include(version)
set_version()

# Options
# =======
OPTION(SCOPI_USE_TBB "enable TBB" OFF)
OPTION(SCOPI_USE_OPENMP "enable OpenMP" OFF)
OPTION(SCOPI_USE_MOSEK "enable Mosek" OFF)
OPTION(SCOPI_USE_SCS "enable SCS" OFF)
OPTION(SCOPI_USE_MKL "enable MKL" OFF)
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
if(SCOPI_USE_TBB)
    set(XTENSOR_USE_TBB 1)
endif()
find_package(xtensor REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
# find_package(plog REQUIRED)
# find_package(nanoflann)

# section needed to use xtensor-blas
# see https://xtensor-blas.readthedocs.io/en/latest/performance.html
add_definitions(-DHAVE_CBLAS=1)

if (WIN32)
    find_package(OpenBLAS REQUIRED)
    set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})
# end of xtensor-blas section

if(SCOPI_USE_TBB)
    find_package(TBB REQUIRED)
endif()

if(SCOPI_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(SCOPI_USE_MKL)
    if(SCOPI_USE_TBB)
        set(MKL_THREAD_LAYER "TBB")
    endif()
    find_package(MKL REQUIRED)
endif()

if(SCOPI_USE_MOSEK)
    find_package(MOSEK REQUIRED)
    if(APPLE)
        set(MOSEK_FUSION_LIBRARY "" CACHE FILEPATH "MOSEK FUSION LIBRARY")
    endif(APPLE)
endif()

if(SCOPI_USE_SCS)
    find_package(scs REQUIRED)
endif()

# Package
# =======

set (CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

set(SCOPI_SRC
    src/contact/base.cpp
    src/contact/contact_brute_force.cpp
    src/contact/contact_kdtree.cpp
    src/problems/DryWithoutFriction.cpp
    src/problems/DryWithFriction.cpp
    src/problems/DryWithFrictionFixedPoint.cpp
    src/problems/ProblemBase.cpp
    src/solvers/ConstraintMosek.cpp
    src/vap/vap_fixed.cpp
    src/vap/vap_fpd.cpp
    src/vap/vap_projection.cpp
    src/minpack.cpp
    src/quaternion.cpp
    src/utils.cpp
)

add_library(scopi ${SCOPI_SRC})

target_include_directories(scopi PUBLIC
                                 $<BUILD_INTERFACE:${SCOPI_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)

target_link_libraries(scopi PUBLIC xtensor
                      nlohmann_json::nlohmann_json
                      fmt::fmt
                      # plog::plog
                      #nanoflann::nanoflann
                      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# set_target_properties(scopi PROPERTIES
#                       PUBLIC_HEADER "${SCOPI_HEADERS}"
#                     #   COMPILE_DEFINITIONS "SCOPI_EXPORTS"
#                       PREFIX ""
#                       VERSION ${${PROJECT_NAME}_VERSION}
#                       SOVERSION ${VERSION_MAJOR}
#                       OUTPUT_NAME "libscopi")

if(SCOPI_USE_MKL)
    target_link_libraries(scopi PUBLIC MKL::MKL)
    target_compile_definitions(scopi PUBLIC SCOPI_USE_MKL)
endif()

if(SCOPI_USE_TBB)
    target_link_libraries(scopi PUBLIC TBB::tbb)
    target_compile_definitions(scopi PUBLIC SCOPI_USE_TBB)
endif()

if(SCOPI_USE_OPENMP)
    target_link_libraries(scopi PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(scopi PUBLIC SCOPI_USE_OPENMP)
endif()

if(SCOPI_USE_MOSEK)
    include_directories(SYSTEM ${MOSEK_INCLUDE_DIR})
    target_link_libraries(scopi LINK_PUBLIC ${MOSEK_LIBRARIES})
    target_include_directories(scopi PUBLIC ${MOSEK_INCLUDE_DIR})
    target_compile_definitions(scopi PUBLIC SCOPI_USE_MOSEK)
endif()

if(SCOPI_USE_SCS)
    include_directories(SYSTEM ${scs_INCLUDE_DIR})
    target_link_libraries(scopi PUBLIC ${scs_LIBRARIES})
    target_include_directories(scopi PUBLIC ${scs_INCLUDE_DIR})
    target_compile_definitions(scopi PUBLIC SCOPI_USE_SCS)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(demos)
else()
    add_subdirectory(demos EXCLUDE_FROM_ALL)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()


# Installation
# ============
include(installation)
install_project()
