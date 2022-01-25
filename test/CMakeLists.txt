include(addGoogleTest)

set(COMMON_BASE
    main.cpp
)

set(SCOPI_TESTS
    test_sphere.cpp
)

foreach(filename IN LISTS SCOPI_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} ${COMMON_BASE} ${filename} ${SCOPI_HEADERS})
    target_include_directories(${targetname} PRIVATE ${SCOPI_INCLUDE_DIR})
    target_link_libraries(${targetname} scopi gtest_main gtest)
endforeach()

# add_executable(test_scopi_lib ${COMMON_BASE} ${SCOPI_TESTS} ${SCOPI_HEADERS})
# target_include_directories(test_scopi_lib PRIVATE ${SCOPI_INCLUDE_DIR})
# # add_dependencies(test_scopi_lib gtest_main gtest)
# target_link_libraries(test_scopi_lib scopi gtest_main gtest)


add_executable(test_adapt test_adapt.cpp)
target_link_libraries(test_adapt scopi)