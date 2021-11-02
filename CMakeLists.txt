cmake_minimum_required(VERSION 3.0)

project(scopi)

# Flags
# ===========

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic ")
set (CMAKE_CXX_STANDARD 17)
message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(SCOPI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${SCOPI_INCLUDE_DIR}/scopi/scopi_config.hpp" scopi_version_defines
     REGEX "#define SCOPI_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${scopi_version_defines})
    if(ver MATCHES "#define SCOPI_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(SCOPI_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(${PROJECT_NAME}_VERSION
    ${SCOPI_VERSION_MAJOR}.${SCOPI_VERSION_MINOR}.${SCOPI_VERSION_PATCH})
message(STATUS "Building scopi v${${PROJECT_NAME}_VERSION}")

# Packages
# ===========

message(STATUS "CMAKE_MODULE_PATH:    " ${CMAKE_MODULE_PATH})

find_package(xtensor REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(MOSEK REQUIRED)
find_package(OpenMP)
find_package(scs REQUIRED)
find_package(absl REQUIRED)
find_package(osqp REQUIRED)

IF (APPLE)
  set(MOSEK_FUSION_LIBRARY "" CACHE FILEPATH "MOSEK FUSION LIBRARY")
endif(APPLE)

message(STATUS "PROJECT_NAME:    " ${PROJECT_NAME})
message(STATUS "CMAKE_INSTALL_NAME_TOOL:    " ${CMAKE_INSTALL_NAME_TOOL})

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

set (CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)

add_subdirectory(include/ecos)
add_library(osqp-cpp STATIC IMPORTED)
set_target_properties(osqp-cpp PROPERTIES IMPORTED_LOCATION /local/home/hbloch/Work/install/osqp-cpp/build/libosqp-cpp.a)
set_target_properties(osqp-cpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES /local/home/hbloch/Work/install/osqp-cpp/include)
target_link_libraries(osqp-cpp INTERFACE osqp::osqp absl::base absl::status)

add_library(scopi_cpp INTERFACE)
include_directories(SYSTEM ${MOSEK_INCLUDE_DIR})
target_include_directories(scopi_cpp INTERFACE $<BUILD_INTERFACE:${SCOPI_INCLUDE_DIR}>
                                               $<INSTALL_INTERFACE:include>
                                               ${ecos_INCLUDE_DIRS_RET})
                                    #  SYSTEM INTERFACE $<BUILD_INTERFACE:${MOSEK_INCLUDE_DIR}>)


# target_link_libraries(scopi_cpp INTERFACE xtensor MKL::MKL ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
# target_link_libraries(scopi_cpp INTERFACE xtensor ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MOSEK_LIBRARY} ${MOSEK_FUSION_LIBRARY})
target_link_libraries(scopi_cpp INTERFACE xtensor nlohmann_json::nlohmann_json fmt::fmt OpenMP::OpenMP_CXX ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MOSEK_LIBRARIES} scs::scsdir osqp::osqp osqp-cpp ecos)

OPTION(BUILD_TESTS "scopi test suite" OFF)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()


# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS scopi_cpp
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${SCOPI_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scopi)

set(SCOPI_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for scopiConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${SCOPI_CMAKECONFIG_INSTALL_DIR})

# scopi is header-only and does not depend on the architecture.
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
# Remove CMAKE_SIZEOF_VOID_P from xtensorConfigVersion.cmake so that an scopiConfig.cmake
set(_SCOPI_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_SCOPI_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${SCOPI_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${SCOPI_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")


############ PYTHON MODULES #################

SET(SCOPI_HEADERS  ${SCOPI_INCLUDE_DIR})
message(STATUS "main ==> SCOPI_HEADERS = ${SCOPI_HEADERS}")

add_subdirectory(demos EXCLUDE_FROM_ALL)

## make VERBOSE=1 pour voir les commandes à la compilation
