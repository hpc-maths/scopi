#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmasked_view.hpp>

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;

    scopi::scopi_container<dim> particles;

    scopi::sphere<dim> s1({{ 0.4, -0.05}}, 0.1);
    scopi::sphere<dim> s2({{ 0.6,  0.05}}, 0.1);
    particles.push_back(s1, scopi::property<dim>().desired_velocity({{-0.25, 0}}).mass(1.).moment_inertia(0.1));
    particles.push_back(s2, scopi::property<dim>().desired_velocity({{0.25, 0}}).mass(1.).moment_inertia(0.1));

    scopi::ScopiSolver<dim> solver(particles, dt);
    solver.run(total_it);

    // scopi::sphere<dim> s1({{0.8, 0.}}, 0.1);

    // auto prop1 = scopi::property<dim>().desired_velocity({{0.25, 0}});

    // particles.push_back(s1, prop1);

    // double dt = 0.1;
    // for(std::size_t nt = 0; nt<100; nt++)
    // {
    //     for (auto& p: particles.pos())
    //     {
    //         if (p[0] > 1.)
    //         {
    //             p[0] -= 1.;
    //         }
    //         else if (p[0] < 0.)
    //         {
    //             p[0] += 1.;
    //         }
    //     }
    //     xt::noalias(particles.v()) = particles.vd();
    //     particles.pos() += dt*particles.v();
    //     std::cout <<  particles.pos() << " " << particles.v() << " " << particles.vd()<< std::endl;
    // }


    // particles.push_back(0, {{1., 1.}});
    // for(std::size_t i = 0; i < particles.size(); ++i)
    // {
    // }
    // std::cout << std::endl;

    // particles.reset_periodic();
    // for(std::size_t i = 0; i < particles.size(); ++i)
    // {
    //     particles[i]->print();
    // }

    return 0;
}
