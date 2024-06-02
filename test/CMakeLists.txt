find_package(doctest REQUIRED)

set(COMMON_BASE
    main.cpp
    utils.cpp
    analytical_solution.cpp
)

set(SCOPI_TESTS
    test_sphere.cpp
    # test_superellipsoid.cpp //need to be checked
    test_closest_points.cpp
    test_container.cpp
    test_contacts_kdtree.cpp
    test_contacts_brute_force.cpp
    test_gradient.cpp
    test_matrices.cpp
    test_obstacles.cpp
    # test_friction.cpp //need to be checked
    # test_viscosity.cpp //need to be checked
    test_quaternions.cpp
    test_worm.cpp
)

foreach(filename IN LISTS SCOPI_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${SCOPI_HEADERS})
    target_include_directories(${targetname} PRIVATE ${SCOPI_INCLUDE_DIR})
    target_compile_features(${targetname} PRIVATE cxx_std_17)
    target_link_libraries(${targetname} scopi doctest::doctest)
endforeach()

add_executable(test_scopi ${COMMON_BASE} ${SCOPI_TESTS})
target_include_directories(test_scopi PRIVATE ${SCOPI_INCLUDE_DIR})
target_compile_features(test_scopi PRIVATE cxx_std_17)
target_link_libraries(test_scopi PRIVATE scopi doctest::doctest)
