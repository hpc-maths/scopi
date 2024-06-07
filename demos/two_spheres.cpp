#include <scopi/container.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/scopi.hpp>
#include <scopi/solver.hpp>

int main(int argc, char** argv)
{
    scopi::initialize("Two spheres simulation");

    constexpr std::size_t dim = 2;
    scopi::sphere<dim> s1(
        {
            {-0.2, -0.05}
    },
        0.1);
    scopi::sphere<dim> s2(
        {
            {0.2, 0.05}
    },
        0.1);

    scopi::scopi_container<dim> particles;
    scopi::ScopiSolver<dim> solver(particles);
    SCOPI_PARSE(argc, argv);

    // particles.push_back(s1,
    //                     scopi::property<dim>()
    //                         .desired_velocity({
    //                             {0.25, 0}
    // })
    //                         .mass(1.)
    //                         .moment_inertia(0.1));
    // particles.push_back(s2,
    //                     scopi::property<dim>()
    //                         .desired_velocity({
    //                             {-0.25, 0}
    // })
    //                         .mass(1.)
    //                         .moment_inertia(0.1));

    // double dt = 0.005;

    // std::size_t total_it = 100;
    // solver.run(dt, total_it);

    return 0;
}
