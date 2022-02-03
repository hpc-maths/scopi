include(addGoogleTest)

set(COMMON_BASE
    main.cpp
)

set(SCOPI_TESTS
    # test_sphere.cpp
    # test_superellipsoid.cpp
    # test_closest_points.cpp
    test_container.cpp
    test_contacts_kdtree.cpp
    # test_contacts_brute_force.cpp
)

foreach(filename IN LISTS SCOPI_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${SCOPI_HEADERS})
    target_include_directories(${targetname} PRIVATE ${SCOPI_INCLUDE_DIR})
    target_link_libraries(${targetname} scopi gtest_main gtest)
endforeach()

add_executable(test_scopi ${COMMON_BASE} ${SCOPI_TESTS})
target_include_directories(test_scopi PRIVATE ${SCOPI_INCLUDE_DIR})
target_link_libraries(test_scopi scopi gtest_main gtest)