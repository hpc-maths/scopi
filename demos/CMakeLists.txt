set(SCOPI_EXAMPLES
   two_ellipsoids.cpp
   two_spheres.cpp
   sphere_plan.cpp
   two_worms.cpp
   box_spheres_3d.cpp
   pile_of_sand_spheres.cpp
   segment.cpp
)

include(generator)
generate_executable(${SCOPI_EXAMPLES})
