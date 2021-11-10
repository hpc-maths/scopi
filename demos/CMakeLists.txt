add_executable(critical_2d critical/2d_case.cpp)
target_link_libraries(critical_2d scopi_cpp)
# if (APPLE)
#   add_custom_command(TARGET critical_2d
#   POST_BUILD COMMAND
#   ${CMAKE_INSTALL_NAME_TOOL} -change
#   `otool -L critical_2d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
#   ${MOSEK_LIBRARIES} critical_2d
#   )
# endif(APPLE)

add_executable(critical_2d_no_overlap critical/2d_case_no_overlap.cpp)
target_link_libraries(critical_2d_no_overlap scopi_cpp)

add_executable(critical_3d critical/3d_case.cpp)
target_link_libraries(critical_3d scopi_cpp)
# if (APPLE)
#   add_custom_command(TARGET critical_3d
#   POST_BUILD COMMAND
#   ${CMAKE_INSTALL_NAME_TOOL} -change
#   `otool -L critical_3d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
#   ${MOSEK_LIBRARIES} critical_3d
#   )
# endif(APPLE)

add_executable(two_spheres two_spheres.cpp)
target_link_libraries(two_spheres scopi_cpp)

add_executable(two_ellipsoids two_ellipsoids.cpp)
target_link_libraries(two_ellipsoids scopi_cpp)
# if (APPLE)
#   add_custom_command(TARGET two_ellipsoid
#   POST_BUILD COMMAND
#   ${CMAKE_INSTALL_NAME_TOOL} -change
#   `otool -L two_ellipsoid | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
#   ${MOSEK_LIBRARIES} two_ellipsoid
#   )
# endif(APPLE)
