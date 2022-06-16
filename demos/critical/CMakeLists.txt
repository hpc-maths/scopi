set(SCOPI_CRITICAL_EXAMPLES
    2d_case.cpp
    2d_case_no_overlap.cpp
    2d_case_no_velocity.cpp
    2d_case_spheres.cpp
    3d_case.cpp
    globules.cpp
)

generate_executable(${SCOPI_CRITICAL_EXAMPLES})
