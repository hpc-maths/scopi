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

    template <class params_t>
    void set_params_test(params_t&)
    {}

//     void set_params_test_uzawa(OptimParamsUzawaBase& params)
//     {
//         params.rho = 200.;
//     }

// #ifdef SCOPI_USE_MKL
//     template <>
//     void set_params_test<OptimUzawaMkl<ViscousWithoutFriction<2>>>(OptimParams<OptimUzawaMkl<ViscousWithoutFriction<2>>>& params)
//     {
//         set_params_test_uzawa(params);
//     }
// #endif
// #ifdef SCOPI_USE_TBB
//     template <>
//     void set_params_test<OptimUzawaMatrixFreeTbb<ViscousWithoutFriction<2>>>(OptimParams<OptimUzawaMatrixFreeTbb<ViscousWithoutFriction<2>>>& params)
//     {
//         set_params_test_uzawa(params);
//     }
// #endif
//     template <>
//     void set_params_test<OptimUzawaMatrixFreeOmp<ViscousWithoutFriction<2>>>(OptimParams<OptimUzawaMatrixFreeOmp<ViscousWithoutFriction<2>>>& params)
//     {
//         set_params_test_uzawa(params);
//     }

    TEST_CASE_TEMPLATE_DEFINE("sphere plan viscosity", SolverType, sphere_plan_viscosity)
    {
        constexpr std::size_t dim = 2;

        double radius = 1.;
        double g = radius;
        double h = 1.5*radius;
        double alpha = PI/4.;
        auto prop = property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);

        double dt = 0.05;

        scopi_container<dim> particles;
        plan<dim> p({{0., 0.}}, PI/2. - alpha);
        sphere<dim> s({{0., h}}, radius);
        particles.push_back(p, property<dim>().deactivate());
        particles.push_back(s, prop.force({{g*std::cos(alpha), -g*std::sin(alpha)}}));


        SolverType solver(particles, dt);
        auto params = solver.get_params();
        set_params_test(params.optim_params);
        params.solver_params.output_frequency = 1;
        solver.run(150);
        particles.f()(1)(1) *= -1.;
        solver.run(189, 150);

        CHECK(diffFile("./Results/scopi_objects_0188.json", "../test/references/sphere_plan_viscosity.json", tolerance));
    }
    TEST_CASE_TEMPLATE_APPLY(sphere_plan_viscosity, solver_dry_without_friction_t<2, >);

    TEST_CASE_TEMPLATE_DEFINE("sphere plan viscosity friction vertical", SolverType, sphere_plan_viscosity_friction_vertical)
    {
        constexpr std::size_t dim = 2;

        double radius = 1.;
        double g = radius;
        double h = 1.5*radius;
        auto prop = property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);

        double dt = 0.05;
        std::size_t total_it = 100;

        scopi_container<dim> particles;
        plan<dim> p({{0., 0.}}, PI/2.);
        sphere<dim> s({{0., h}}, radius);
        particles.push_back(p, property<dim>().deactivate());
        particles.push_back(s, prop.force({{0, -g}}));

        SolverType solver(particles, dt);
        auto params = solver.get_params();
        // params.problem_params.mu = 0.1;
        params.solver_params.output_frequency = 1;
        solver.run(total_it);
        particles.f()(1)(1) *= -1.;
        solver.run(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity_friction_vertical.json", tolerance));
    }
    TEST_CASE_TEMPLATE_APPLY(sphere_plan_viscosity_friction_vertical, solver_dry_with_friction_t<2, vap_fpd>);

    TEST_CASE_TEMPLATE_DEFINE("sphere plan viscosity friction", SolverType, sphere_plan_viscosity_friction)
    {
        constexpr std::size_t dim = 2;

        double radius = 1.;
        double g = radius;
        double h = 1.5*radius;
        auto prop = property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);

        double dt = 0.05;
        std::size_t total_it = 100;

        scopi_container<dim> particles;
        plan<dim> p({{0., 0.}}, PI/2.);
        sphere<dim> s({{0., h}}, radius);
        particles.push_back(p, property<dim>().deactivate());
        particles.push_back(s, prop.force({{g, -g}}));

        SolverType solver(particles, dt);
        // auto params = solver.get_params();
        // params.problem_params.mu = 0.1;
        solver.run(total_it);
        particles.f()(1)(1) *= -1.;
        solver.run(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity_friction.json", tolerance));
    }
    TEST_CASE_TEMPLATE_APPLY(sphere_plan_viscosity_friction, solver_dry_with_friction_t<2, vap_fpd>);

}
