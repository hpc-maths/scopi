set(SCOPI_PROCEEDING_MULTIPARTICLES
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
)

generate_executable(${SCOPI_PROCEEDING_MULTIPARTICLES})
