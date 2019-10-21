set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_STANDARD 17)

add_executable(test_soa test_soa.cpp ${SCOPI_HEADERS})
target_link_libraries(test_soa scopi_cpp)

add_executable(test_soa_xtensor test_soa_xtensor.cpp ${SCOPI_HEADERS})
target_link_libraries(test_soa_xtensor scopi_cpp)

add_executable(test_nanoflann test_nanoflann.cpp ${SCOPI_HEADERS})
target_link_libraries(test_nanoflann scopi_cpp)

add_executable(test_uzawa test_uzawa.cpp ${SCOPI_HEADERS})
target_link_libraries(test_uzawa scopi_cpp)
