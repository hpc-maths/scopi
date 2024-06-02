#include <scopi/objects/types/sphere.hpp>
#include <scopi/problems/ViscousWithFriction.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <xtensor/xmath.hpp>

int main()
{
    plog::init(plog::info, "two_spheres_viscosity.log");

    constexpr std::size_t dim = 2;

    double dt            = .001;
    std::size_t total_it = 3000;

    double r = 0.1;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(1. * r * r / 2.);

    scopi::sphere<dim> s1(
        {
            {0., 0.}
    },
        r);
    scopi::sphere<dim> s2(
        {
            {1., 0.05}
    },
        r);
    scopi::sphere<dim> s3(
        {
            {0., 0.3}
    },
        r);
    scopi::sphere<dim> s4(
        {
            {1., 0.35}
    },
        r);
    // particles.push_back(s1, prop.velocity({1., 0.}));
    // particles.push_back(s1, scopi::property<dim>().deactivate());
    // particles.push_back(s2, prop.velocity({-1., 0.}));
    particles.push_back(s1,
                        prop.force({
                            {1, 0}
    }));
    particles.push_back(s2,
                        prop.force({
                            {-1, 0}
    }));
    particles.push_back(s3,
                        prop.force({
                            {1, 0}
    }));
    particles.push_back(s4,
                        prop.force({
                            {-1, 0}
    }));

    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::ViscousWithFriction<dim>>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
    auto params                                  = solver.get_params();
    params.optim_params.change_default_tol_mosek = false;
    params.problem_params.mu                     = 0.1;
    params.contact_params.dmax                   = 0.11;
    params.problem_params.gamma_min              = -0.7;
    // params.problem_params.tol = 10^(-8);
    params.solver_params.output_frequency = 1;
    params.solver_params.path             = "_two_spheres";
    params.solver_params.filename         = "two_spheres";
    solver.run(total_it);

    return 0;
}
