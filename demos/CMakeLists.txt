set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_STANDARD 17)

# add_executable(test_scopi test_scopi.cpp ${SCOPI_HEADERS})
# target_link_libraries(test_scopi scopi_cpp_lib MKL::MKL ${VTK_LIBRARIES})

add_executable(test_container test_container.cpp)
target_link_libraries(test_container scopi_cpp)

add_executable(test_dispatch test_dispatch.cpp)
target_link_libraries(test_dispatch scopi_cpp)


add_executable(test_distance test_distance.cpp)
target_link_libraries(test_distance scopi_cpp )
if (APPLE)
  add_custom_command(TARGET test_distance
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L test_distance | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARIES} test_distance
  )
endif(APPLE)

add_executable(test_distance_3d test_distance_3d.cpp)
target_link_libraries(test_distance_3d scopi_cpp)
if (APPLE)
  add_custom_command(TARGET test_distance_3d
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L test_distance_3d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARIES} test_distance_3d
  )
endif(APPLE)

add_executable(test_mosek test_mosek.cpp)
target_link_libraries(test_mosek scopi_cpp)
if (APPLE)
  add_custom_command(TARGET test_mosek
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L test_mosek | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARIES} test_mosek
  )
endif(APPLE)

add_executable(critical_2d critical/2d_case.cpp)
target_link_libraries(critical_2d scopi_cpp)
if (APPLE)
  add_custom_command(TARGET critical_2d
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L critical_2d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARIES} critical_2d
  )
endif(APPLE)

add_executable(critical_3d critical/3d_case.cpp)
target_link_libraries(critical_3d scopi_cpp)
if (APPLE)
  add_custom_command(TARGET critical_3d
  POST_BUILD COMMAND
  ${CMAKE_INSTALL_NAME_TOOL} -change
  `otool -L critical_3d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
  ${MOSEK_LIBRARIES} critical_3d
  )
endif(APPLE)
