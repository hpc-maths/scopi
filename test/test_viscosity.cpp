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

    template <class OptimParamsType>
    void set_params_test(OptimParamsType&)
    {}

    template <class OptimParamsType>
    void set_params_test_uzawa(OptimParamsType& params)
    {
        params.m_rho = 200.;
    }

#ifdef SCOPI_USE_MKL
    template <>
    void set_params_test<OptimParams<OptimUzawaMkl>>(OptimParams<OptimUzawaMkl>& params)
    {
        set_params_test_uzawa(params);
    }
#endif
    template <>
    void set_params_test<OptimParams<OptimUzawaMatrixFreeTbb>>(OptimParams<OptimUzawaMatrixFreeTbb>& params)
    {
        set_params_test_uzawa(params);
    }
    template <>
    void set_params_test<OptimParams<OptimUzawaMatrixFreeOmp>>(OptimParams<OptimUzawaMatrixFreeOmp>& params)
    {
        set_params_test_uzawa(params);
    }

    TEST_CASE_TEMPLATE("sphere plan viscosity", SolverAndParams, SOLVER_VISCOUS_WITHOUT_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_VISCOUS_WITHOUT_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using SolverType = typename SolverAndParams::SolverType;
        using OptimParamsType = typename SolverAndParams::OptimParamsType;
        using ProblemParamsType = typename SolverAndParams::ProblemParamsType;

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

        OptimParamsType optim_params;
        set_params_test(optim_params);
        ProblemParamsType problem_params;

        SolverType solver(particles, dt, optim_params, problem_params);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity.json", tolerance));
    }

    TEST_CASE_TEMPLATE("sphere plan viscosity friction vertical", SolverAndParams, SOLVER_VISCOUS_WITH_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_VISCOUS_WITH_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using SolverType = typename SolverAndParams::SolverType;
        using OptimParamsType = typename SolverAndParams::OptimParamsType;
        using ProblemParamsType = typename SolverAndParams::ProblemParamsType;

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
        particles.push_back(s, prop.force({{0, -g}}));

        OptimParamsType optim_params;
        ProblemParamsType problem_params;
        problem_params.m_mu = 0.1;
        SolverType solver(particles, dt, optim_params, problem_params);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity_friction_vertical.json", tolerance));
    }

    TEST_CASE_TEMPLATE("sphere plan viscosity friction", SolverAndParams, SOLVER_VISCOUS_WITH_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_VISCOUS_WITH_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using SolverType = typename SolverAndParams::SolverType;
        using OptimParamsType = typename SolverAndParams::OptimParamsType;
        using ProblemParamsType = typename SolverAndParams::ProblemParamsType;

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

        OptimParamsType optim_params;
        ProblemParamsType problem_params;
        problem_params.m_mu = 0.1;
        SolverType solver(particles, dt, optim_params, problem_params);
        solver.solve(total_it);
        particles.f()(1)(1) *= -1.;
        solver.solve(2*total_it, total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/sphere_plan_viscosity_friction.json", tolerance));
    }

}
