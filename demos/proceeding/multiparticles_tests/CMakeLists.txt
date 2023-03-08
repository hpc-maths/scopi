list(APPEND SCOPI_PROCEEDING_MULTIPARTICLES
    pile_of_sand_friction_small_config_mu01_convex.cpp
    pile_of_sand_friction_small_config_mu01_fixed_point.cpp
    pile_of_sand_friction_small_config_mu1_convex.cpp
    pile_of_sand_friction_small_config_mu1_fixed_point.cpp
    pile_of_sand_friction_large_config_mu01_convex.cpp
    pile_of_sand_friction_large_config_mu01_fixed_point.cpp
    pile_of_sand_friction_large_config_mu1_convex.cpp
    pile_of_sand_friction_large_config_mu1_fixed_point.cpp
    pile_of_sand_friction_figure_mu01_convex.cpp
    pile_of_sand_friction_figure_mu01_fixed_point.cpp
    pile_of_sand_friction_figure_mu1_convex.cpp
    pile_of_sand_friction_figure_mu1_fixed_point.cpp
)

if(SCOPI_USE_MKL)
    list(APPEND SCOPI_PROCEEDING_MULTIPARTICLES
         spheres_in_box_small_config_pgd.cpp
         spheres_in_box_small_config_apgd.cpp
         spheres_in_box_small_config_apgdas.cpp
         spheres_in_box_small_config_apgdar.cpp
         spheres_in_box_small_config_apgdasr.cpp
         spheres_in_box_large_config_pgd.cpp
         spheres_in_box_large_config_apgd.cpp
         spheres_in_box_large_config_apgdas.cpp
         spheres_in_box_large_config_apgdar.cpp
         spheres_in_box_large_config_apgdasr.cpp
         pile_of_sand_ellipses.cpp
         )
endif()

generate_executable(${SCOPI_PROCEEDING_MULTIPARTICLES})
