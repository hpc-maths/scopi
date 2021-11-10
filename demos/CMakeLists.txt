cmake_minimum_required(VERSION 3.6)
project(Scopi)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_STANDARD 17)


add_executable(critical_2d critical/2d_case.cpp)
target_link_libraries(critical_2d scopi_cpp)
if (APPLE)
  add_custom_command(TARGET critical_2d
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L critical_2d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARY} critical_2d
  )
endif(APPLE)

add_executable(critical_3d critical/3d_case.cpp)
target_link_libraries(critical_3d scopi_cpp)
if (APPLE)
  add_custom_command(TARGET critical_3d
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L critical_3d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARY} critical_3d
  )
endif(APPLE)

add_executable(two_spheres two_spheres.cpp)
target_link_libraries(two_spheres scopi_cpp)
if (APPLE)
   add_custom_command(TARGET two_spheres
   POST_BUILD COMMAND
   ${CMAKE_INSTALL_NAME_TOOL} -change
   `otool -L two_spheres | sed -n -e \" "s/.*\\(libmosek.*dylib\\).*/\\1/p" \"`
   ${MOSEK_LIBRARY} two_spheres
   )
endif(APPLE)

add_executable(two_ellipsoids two_ellipsoids.cpp)
target_link_libraries(two_ellipsoids scopi_cpp)
if (APPLE)
   add_custom_command(TARGET two_ellipsoids
   POST_BUILD COMMAND
   ${CMAKE_INSTALL_NAME_TOOL} -change
   `otool -L two_ellipsoids | sed -n -e \" "s/.*\\(libmosek.*dylib\\).*/\\1/p" \"`
   ${MOSEK_LIBRARY} two_ellipsoids
   )
endif(APPLE)
