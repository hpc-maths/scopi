set(SCOPI_EXAMPLES
   two_ellipsoids.cpp
   two_ellipsoids_asymmetrical_2d.cpp
   two_ellipsoids_asymmetrical_3d.cpp
   two_ellipsoids_spheres_2d.cpp
   two_ellipsoids_spheres_3d.cpp
   two_spheres.cpp
   two_spheres_periodic.cpp
   sphere_plan.cpp
   viscosity.cpp
   two_spheres_viscosity.cpp
   two_worms.cpp
   three_worms_periodic.cpp
   convergence_scheme.cpp
   box_spheres_3d.cpp
   pile_of_sand_spheres.cpp
   pile_of_sand_superellipsoids.cpp
)

include(generator)
generate_executable(${SCOPI_EXAMPLES})

add_subdirectory(critical)
add_subdirectory(proceeding)
