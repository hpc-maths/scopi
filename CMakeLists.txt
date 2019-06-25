cmake_minimum_required(VERSION 3.0)

project(scopi)

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

set(SCOPI_HEADERS
    ${SCOPI_INCLUDE_DIR}/scopi/nanoflann.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/scopi_config.hpp
    ${SCOPI_INCLUDE_DIR}/scopi/soa.hpp
)

#find_package(HDF5)
#if(HDF5_FOUND)
#    message("HDF5_C_LIBRARY_hdf5 : ${HDF5_C_LIBRARY_hdf5}")
#    ##include_directories(${HDF5_INCLUDE_DIR})
#    ##set(_hdf5_libs hdf5 hdf5_cpp)
#endif(HDF5_FOUND)


add_library(scopi INTERFACE)
target_include_directories(scopi INTERFACE $<BUILD_INTERFACE:${SCOPI_INCLUDE_DIR}>
                                          $<INSTALL_INTERFACE:include>)
target_link_libraries(scopi INTERFACE ${HDF5_C_LIBRARY_hdf5})

OPTION(BUILD_TESTS "scopi test suite" OFF)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

add_subdirectory(demos)

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS scopi
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
