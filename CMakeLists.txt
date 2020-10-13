cmake_minimum_required(VERSION 3.0)

project(scopi)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_STANDARD 17)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

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

find_package(xtensor REQUIRED)

set(MKL_THREAD_LAYER "TBB")
find_package(MKL)

#set(SCOPI_HEADERS
#    ${SCOPI_INCLUDE_DIR}/scopi/nanoflann.hpp
#    ${SCOPI_INCLUDE_DIR}/scopi/scopi_config.hpp
#    ${SCOPI_INCLUDE_DIR}/scopi/soa.hpp
#)

add_library(scopi_cpp INTERFACE)
target_include_directories(scopi_cpp INTERFACE $<BUILD_INTERFACE:${SCOPI_INCLUDE_DIR}>
                                          $<INSTALL_INTERFACE:include>)
target_link_libraries(scopi_cpp INTERFACE xtensor MKL::MKL)

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

# find_package (Python3 COMPONENTS Interpreter Development NumPy)
# message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
# message(STATUS "PYTHON_LIBRARY=${PYTHON_LIBRARY}")
# message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
# message(STATUS "Python3_INCLUDE_DIR=${Python3_INCLUDE_DIR}")
# message(STATUS "Python3_LIBRARY_RELEASE=${Python3_LIBRARY_RELEASE}")
# message(STATUS "Python3_INCLUDE_DIR=${Python3_INCLUDE_DIR}")
# message(STATUS "Python3_NumPy_INCLUDE_DIR=${Python3_NumPy_INCLUDE_DIR}")
# message(STATUS "==> Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
# message(STATUS "==> Python3_NumPy_INCLUDE_DIRS=${Python3_NumPy_INCLUDE_DIRS}")

# find_package(pybind11 REQUIRED)
# set(PYBIND11_INCLUDE_DIR ${pybind11_INCLUDE_DIRS})
# message(STATUS "PYBIND11_MODULE=${PYBIND11_MODULE}")
# message(STATUS "PYBIND11_INCLUDE_DIR=${PYBIND11_INCLUDE_DIR}")
# #message(STATUS "pybind11 module=${pybind11::module}")

# find_package(xtensor-python REQUIRED CONFIG)
# set(XTENSOR_PYTHON_INCLUDE_DIR ${xtensor-python_INCLUDE_DIRS})
# message(STATUS "XTENSOR_PYTHON_INCLUDE_DIR=${XTENSOR_PYTHON_INCLUDE_DIR}")

# ### Gestion de VTK
# ##OPTION(WITH_VTK "Set to OFF to build without VTK" ON )
# ##IF (WITH_VTK)
# FIND_PACKAGE(VTK)
# IF (VTK_FOUND)
#   INCLUDE (${VTK_USE_FILE})
#   message(STATUS "VTK_USE_FILE=${VTK_USE_FILE}")
#   message(STATUS "VTK_LIBRARIES=${VTK_LIBRARIES}")
# ENDIF (VTK_FOUND)
# ##ENDIF (WITH_VTK)



SET(SCOPI_HEADERS  ${SCOPI_INCLUDE_DIR})
message(STATUS "main ==> SCOPI_HEADERS = ${SCOPI_HEADERS}")


SET(scopi_src
src/fm/min_heap.cpp
src/fm/fm.cpp
src/scopi/particles.cpp
src/scopi/contacts.cpp
src/scopi/projection.cpp
src/scopi/obstacle.cpp
src/scopi/obstacles.cpp
)
# ADD_LIBRARY(scopi_cpp_lib	STATIC ${scopi_src})
# target_include_directories(scopi_cpp_lib PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} include/ ${XTENSOR_PYTHON_INCLUDE_DIR} ${MKL_INCLUDE_DIR} ${VTK_USE_FILE})

add_subdirectory(demos)


# ### Module "fm"
# add_library(fm MODULE src/fm/fm.cpp src/fm/min_heap.cpp)
# target_include_directories(fm PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} include/ ${XTENSOR_PYTHON_INCLUDE_DIR})
# target_link_libraries(fm PRIVATE pybind11::module)
# set_target_properties(fm PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
#                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")
# ### Module "scopi"
# add_library(scopi MODULE src/fm/min_heap.cpp src/fm/fm.cpp src/scopi/particles.cpp src/scopi/contacts.cpp src/scopi/projection.cpp src/scopi/scopi.cpp src/scopi/obstacle.cpp src/scopi/obstacles.cpp)
# target_include_directories(scopi PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} include/ ${XTENSOR_PYTHON_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
# target_link_libraries(scopi INTERFACE xtensor  PRIVATE MKL::MKL pybind11::module )
# set_target_properties(scopi PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
#                                     SUFFIX "${PYTHON_MODULE_EXTENSION}")



## make VERBOSE=1 pour voir les commandes à la compilation
