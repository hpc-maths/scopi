#include <cstddef>
#include "doctest/doctest.h"
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>
#include <tuple>

namespace scopi {

    TEST_CASE_TEMPLATE("sphere plan viscosity", SolverType, SOLVER_WITH_CONTACT_VISCOSITY(2, contact_kdtree, vap_fpd), SOLVER_WITH_CONTACT_VISCOSITY(2, contact_brute_force, vap_fpd))
    {
        constexpr std::size_t dim = 2;
        double PI = xt::numeric_constants<double>::PI;

        double radius = 1.;
        double g = radius;
        double h = 1.5*radius;
        auto prop = scopi::property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);

        double dt = 0.05;
        std::size_t total_it = 100;

        scopi::scopi_container<dim> particles;
        scopi::plan<dim> p({{0., 0.}}, PI/2.);
        scopi::sphere<dim> s({{0., h}}, radius);
        particles.push_back(p, scopi::property<dim>().deactivate());
        particles.push_back(s, prop.force({{g, -g}}));

        scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::MatrixOptimSolverViscosity<dim>>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity.json", tolerance));
    }

}
