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
find_package(xtensor REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
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
    src/solvers/MatrixOptimSolver.cpp
    src/solvers/MatrixOptimSolverFriction.cpp
    src/solvers/OptimScs.cpp
    src/solvers/OptimUzawaMatrixFreeOmp.cpp
    src/solvers/OptimUzawaMatrixFreeTbb.cpp
    src/solvers/OptimUzawaMkl.cpp
    src/vap/vap_fixed.cpp
    src/vap/vap_fpd.cpp
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
                                        #   nanoflann::nanoflann
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
    target_link_libraries(scopi PUBLIC ${MOSEK_LIBRARIES})
    target_compile_definitions(scopi PUBLIC SCOPI_USE_MOSEK)
endif()

if(SCOPI_USE_SCS)
    include_directories(SYSTEM ${scs_INCLUDE_DIR})
    target_link_libraries(scopi PUBLIC ${scs_LIBRARIES})
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

set(SCOPI_HEADERS
    ${SCOPI_INCLUDE_DIR}/scopi/container.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/contact/contact_brute_force.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/contact/base.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/contact/contact_kdtree.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solver.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/dispatch.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/neighbor.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/types/sphere.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/types/globule.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/types/superellipsoid.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/types/constructor.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/types/base.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/types/plan.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/methods/write_objects.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/methods/closest_points.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/objects/methods/distance.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimMosek.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimUzawaMatrixFreeOmp.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimScs.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/MatrixOptimSolver.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/MatrixOptimSolverFriction.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimUzawaMatrixFreeTbb.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimUzawaMkl.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimUzawaBase.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/solvers/OptimBase.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/quaternion.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/minpack.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/property.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/crtp.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/utils.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/types.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/scopi_config.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/vap/vap_fpd.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/vap/base.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/vap/vap_fixed.hpp
)

include(installation)
install_project()
