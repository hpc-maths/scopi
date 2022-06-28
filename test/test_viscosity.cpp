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

    template <class solver_t>
    void set_params_test(OptimParams<solver_t>&)
    {}

    void set_params_test_uzawa(OptimParamsUzawaBase<ViscousWithoutFriction<2>>& params)
    {
        params.rho = 200.;
    }

#ifdef SCOPI_USE_MKL
    template <>
    void set_params_test<OptimUzawaMkl<ViscousWithoutFriction<2>>>(OptimParams<OptimUzawaMkl<ViscousWithoutFriction<2>>>& params)
    {
        set_params_test_uzawa(params);
    }
#endif
    template <>
    void set_params_test<OptimUzawaMatrixFreeTbb<ViscousWithoutFriction<2>>>(OptimParams<OptimUzawaMatrixFreeTbb<ViscousWithoutFriction<2>>>& params)
    {
        set_params_test_uzawa(params);
    }
    template <>
    void set_params_test<OptimUzawaMatrixFreeOmp<ViscousWithoutFriction<2>>>(OptimParams<OptimUzawaMatrixFreeOmp<ViscousWithoutFriction<2>>>& params)
    {
        set_params_test_uzawa(params);
    }

    TEST_CASE_TEMPLATE("sphere plan viscosity", SolverType, SOLVER_VISCOUS_WITHOUT_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_VISCOUS_WITHOUT_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using solver_t = typename SolverType::solver_type;
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

        OptimParams<solver_t> params;
        set_params_test(params);

        SolverType solver(particles, dt, params);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity.json", tolerance));
    }

    TEST_CASE_TEMPLATE("sphere plan viscosity friction vertical", SolverType, SOLVER_VISCOUS_WITH_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_VISCOUS_WITH_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using solver_t = typename SolverType::solver_type;
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

        OptimParams<solver_t> params;
        params.problem_params.mu = 0.1;
        SolverType solver(particles, dt, params);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity_friction_vertical.json", tolerance));
    }

    TEST_CASE_TEMPLATE("sphere plan viscosity friction", SolverType, SOLVER_VISCOUS_WITH_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_VISCOUS_WITH_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using solver_t = typename SolverType::solver_type;
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

        OptimParams<solver_t> params;
        params.problem_params.mu = 0.1;
        SolverType solver(particles, dt, params);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity_friction.json", tolerance));
    }

}
