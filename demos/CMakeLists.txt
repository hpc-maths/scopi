set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_STANDARD 17)

add_executable(test_soa test_soa.cpp ${SCOPI_HEADERS})
target_link_libraries(test_soa scopi)