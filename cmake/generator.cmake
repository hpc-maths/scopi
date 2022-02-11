function (generate_executable)
    message(${ARGN})
    foreach(filename IN LISTS ARGN)
        string(REPLACE ".cpp" "" targetname ${filename})
        add_executable(${targetname} ${filename})
        target_link_libraries(${targetname} scopi)

        # if (APPLE)
        #   add_custom_command(TARGET critical_2d
        #   POST_BUILD COMMAND
        #   ${CMAKE_INSTALL_NAME_TOOL} -change
        #   `otool -L critical_2d | sed -n -e \""s/.*\\(libmosek.*dylib\\).*/\\1/p"\"`
        #   ${MOSEK_LIBRARY} critical_2d
        #   )
        # endif(APPLE)

    endforeach()
endfunction()