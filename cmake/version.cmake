function (set_version)
    string(TOLOWER ${PROJECT_NAME} PROJECT_LOW)
    string(TOUPPER ${PROJECT_NAME} PROJECT_CAP)
    file(STRINGS "${${PROJECT_CAP}_INCLUDE_DIR}/${PROJECT_LOW}/${PROJECT_LOW}_config.hpp" version_defines
        REGEX "#define ${PROJECT_CAP}_VERSION_(MAJOR|MINOR|PATCH)")

    foreach(ver ${version_defines})
        if(ver MATCHES "#define ${PROJECT_CAP}_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
            set(VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
        endif()
    endforeach()

    # set(PROJECT_VERSION
    #     ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH} CACHE
    #     STRING "Version of the project" FORCE)
    set(PROJECT_VERSION
        ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH} PARENT_SCOPE)
    message(STATUS "Building ${PROJECT_LOW} v${PROJECT_VERSION}")
endfunction()