set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_STANDARD 17)

# add_executable(test_scopi test_scopi.cpp ${SCOPI_HEADERS})
# target_link_libraries(test_scopi scopi_cpp_lib MKL::MKL ${VTK_LIBRARIES})

add_executable(test_container test_container.cpp)
target_link_libraries(test_container scopi_cpp)
