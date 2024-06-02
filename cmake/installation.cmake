function (install_project)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    string(TOLOWER ${PROJECT_NAME} PROJECT_LOW)
    string(TOUPPER ${PROJECT_NAME} PROJECT_CAP)

    install(TARGETS ${PROJECT_LOW}
            EXPORT ${PROJECT_NAME}-targets)

    # Makes the project importable from the build directory
    export(EXPORT ${PROJECT_NAME}-targets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

    install(DIRECTORY ${${PROJECT_CAP}_INCLUDE_DIR}/${PROJECT_LOW}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    set(${PROJECT_CAP}_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
        STRING "install path for ${PROJECT_NAME}Config.cmake")

    configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                INSTALL_DESTINATION ${${PROJECT_CAP}_CMAKECONFIG_INSTALL_DIR})

    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                    VERSION ${PROJECT_VERSION}
                                    COMPATIBILITY AnyNewerVersion
                                    ARCH_INDEPENDENT)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
            DESTINATION ${${PROJECT_CAP}_CMAKECONFIG_INSTALL_DIR})
    install(EXPORT ${PROJECT_NAME}-targets
            FILE ${PROJECT_NAME}Targets.cmake
            DESTINATION ${${PROJECT_CAP}_CMAKECONFIG_INSTALL_DIR})

    configure_file(${PROJECT_NAME}.pc.in
                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                    @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
endfunction()
