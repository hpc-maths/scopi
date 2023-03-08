list(APPEND SCOPI_PROCEEDING_DISK_ON_INCLINED_PLANE
    distance_and_velocity_with_friction_mu01_convex.cpp
    distance_and_velocity_with_friction_mu01_fixedPoint.cpp
    distance_and_velocity_with_friction_mu1_convex.cpp
    distance_and_velocity_with_friction_mu1_fixedPoint.cpp
    error_with_friction_mu01_convex.cpp
    error_with_friction_mu01_fixedPoint.cpp
    error_with_friction_mu1_convex.cpp
    error_with_friction_mu1_fixedPoint.cpp
)

if(SCOPI_USE_MKL)
    list(APPEND SCOPI_PROCEEDING_DISK_ON_INCLINED_PLANE
         distance_and_velocity_without_friction.cpp
         oneStep_pgd_smallTol.cpp
         oneStep_apgd_smallTol.cpp
         oneStep_apgdas_smallTol.cpp
         oneStep_apgdar_smallTol.cpp
         oneStep_apgdasr_smallTol.cpp
         oneStep_pgd_largeTol.cpp
         oneStep_apgd_largeTol.cpp
         oneStep_apgdas_largeTol.cpp
         oneStep_apgdar_largeTol.cpp
         oneStep_apgdasr_largeTol.cpp
         error_without_friction.cpp
    )
endif()

generate_executable(${SCOPI_PROCEEDING_DISK_ON_INCLINED_PLANE})
