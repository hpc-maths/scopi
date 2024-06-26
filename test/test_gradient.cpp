#include <doctest/doctest.h>

#include <scopi/container.hpp>
#include <scopi/objects/types/plane.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/solvers/OptimGradient.hpp>
#include <scopi/solvers/apgd.hpp>

#include "analytical_solution.hpp"
#include "utils.hpp"

namespace scopi
{
    double tol            = 1e-5;
    double tol_analytical = 1e-12;

    /// TESTS SPHERES - PLANE
    TEST_CASE("sphere - plane without friction")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        double radius        = 1.;
        double g             = 1.;
        double dt            = 0.01;
        std::size_t total_it = 1000;
        double h             = 2. * radius;
        double alpha         = PI / 4;
        plane<dim> plane(
            {
                {0., 0.}
        },
            PI / 2);
        sphere<dim> sphere(
            {
                {0, h}
        },
            radius);
        particles.push_back(plane, property<dim>().deactivate());
        particles.push_back(sphere,
                            property<dim>()
                                .mass(1)
                                .moment_inertia(1. * radius * radius / 2.)
                                .force({
                                    {g * sin(alpha), -g * cos(alpha)}
        }));

        using problem_t    = NoFriction;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);
        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.01;
        params.optim_params.dynamic_descent = true;

        solver.run(dt, total_it);

        auto pos            = particles.pos();
        auto q              = particles.q();
        auto tmp            = analytical_solution_sphere_plane_no_friction(alpha, dt * (total_it), radius, g, h);
        auto pos_analytical = tmp.first;
        auto q_analytical   = quaternion(tmp.second);
        double error_pos    = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        double error_q      = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);
        auto v              = particles.v();
        auto omega          = particles.omega();
        tmp                 = analytical_solution_sphere_plane_velocity_no_friction(alpha, dt * (total_it), radius, g, h);
        auto v_analytical   = tmp.first;
        double error_v      = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);

        REQUIRE(error_pos <= doctest::Approx(1e-3));
        REQUIRE(error_q <= doctest::Approx(1e-3));
        REQUIRE(error_v <= doctest::Approx(tol_analytical));
        REQUIRE(omega(1) == doctest::Approx(0.));
    }

    TEST_CASE("sphere - plane viscous without friction")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        double radius        = 1.;
        double g             = 1.;
        double dt            = 0.01;
        std::size_t total_it = 500;
        double h             = 2. * radius;
        double alpha         = PI / 4;
        plane<dim> plane(
            {
                {0., 0.}
        },
            PI / 2);
        sphere<dim> sphere(
            {
                {0, h}
        },
            radius);
        particles.push_back(plane, property<dim>().deactivate());
        particles.push_back(sphere,
                            property<dim>()
                                .mass(1)
                                .moment_inertia(1. * radius * radius / 2.)
                                .force({
                                    {g * sin(alpha), -g * cos(alpha)}
        }));

        using problem_t    = Viscous;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);
        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        // double gamma     = 0;
        double gamma_min = -2.;
        // double gamma_tol = 1e-6;

        solver.run(dt, total_it);
        particles.f()(1)  = {-g * sin(alpha), g * cos(alpha)};
        double total_it_2 = 500;
        solver.run(dt, total_it_2);

        auto pos = particles.pos();
        auto q   = particles.q();
        auto tmp = analytical_solution_sphere_plane_viscous(alpha, dt * (total_it + total_it_2), radius, g, h, gamma_min, dt * (total_it));
        auto pos_analytical = tmp.first;
        auto q_analytical   = quaternion(tmp.second);
        double error_pos    = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        double error_q      = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);
        auto v              = particles.v();
        auto omega          = particles.omega();
        tmp = analytical_solution_sphere_plane_velocity_viscous(alpha, dt * (total_it + total_it_2), radius, g, h, gamma_min, dt * (total_it));
        auto v_analytical = tmp.first;
        double error_v    = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);

        REQUIRE(error_pos <= doctest::Approx(1e-3));
        REQUIRE(error_q <= doctest::Approx(1e-3));
        REQUIRE(error_v <= doctest::Approx(1e-3));
        REQUIRE(omega(1) == doctest::Approx(0.));
    }

    TEST_CASE("sphere - plane friction no fixed point")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        plane<dim> plane(
            {
                {0., 0.}
        },
            PI / 2 - PI / 6);

        double dd = 1;
        double rr = 2;
        sphere<dim> sphere(
            {
                {0, (rr + dd) / std::cos(PI / 6)}
        },
            rr);

        particles.push_back(plane, property<dim>().deactivate());
        particles.push_back(sphere,
                            property<dim>()
                                .mass(1)
                                .moment_inertia(0.5 * rr * rr)
                                .force({
                                    {0, -1}
        }));

        double Tf = 20;
        double dt = 0.1;

        using problem_t    = Friction;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);

        fs::path path        = "test_gradient";
        std::string filename = "sphere_plane_friction_mu01";

        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        std::size_t total_it                  = Tf / dt;
        params.solver_params.output_frequency = total_it;
        params.solver_params.path             = path;
        params.solver_params.filename         = filename;

        // mu01
        params.default_contact_property.mu = 0.1;
        solver.run(dt, total_it);

        CHECK(check_reference_file(path, filename, total_it, tol));
    }

    TEST_CASE("sphere - plane friction fixed point")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        double radius        = 1.;
        double g             = 1.;
        double dt            = 0.01;
        std::size_t total_it = 500;
        double h             = 2. * radius;
        double alpha         = PI / 4;
        plane<dim> plane(
            {
                {0., 0.}
        },
            PI / 2);
        sphere<dim> sphere(
            {
                {0, h}
        },
            radius);
        particles.push_back(plane, property<dim>().deactivate());
        particles.push_back(sphere,
                            property<dim>()
                                .mass(1)
                                .moment_inertia(1. * radius * radius / 2.)
                                .force({
                                    {g * sin(alpha), -g * cos(alpha)}
        }));

        using problem_t    = FrictionFixedPoint;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);

        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.01;
        params.optim_params.dynamic_descent = true;

        double mu                                            = 0.5;
        params.default_contact_property.mu                   = mu;
        params.default_contact_property.fixed_point_tol      = 1e-6;
        params.default_contact_property.fixed_point_max_iter = 1000;
        solver.run(dt, total_it);

        auto pos              = particles.pos();
        auto q                = particles.q();
        auto tmp              = analytical_solution_sphere_plane_friction(alpha, mu, dt * (total_it), radius, g, h);
        auto pos_analytical   = tmp.first;
        auto q_analytical     = quaternion(tmp.second);
        double error_pos      = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        double error_q        = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);
        auto v                = particles.v();
        auto omega            = particles.omega();
        tmp                   = analytical_solution_sphere_plane_velocity_friction(alpha, mu, dt * (total_it), radius, g, h);
        auto v_analytical     = tmp.first;
        auto omega_analytical = tmp.second;
        double error_v        = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);
        double error_omega    = std::norm(omega(1) - omega_analytical) / std::norm(omega_analytical);
        REQUIRE(error_pos <= doctest::Approx(1e-2));
        REQUIRE(error_q <= doctest::Approx(1e-2));
        REQUIRE(error_v <= doctest::Approx(tol_analytical));
        REQUIRE(error_omega <= doctest::Approx(tol_analytical));
    }

    TEST_CASE("sphere - plane viscous friction")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        double radius        = 1.;
        double g             = 1.;
        double dt            = 0.01;
        std::size_t total_it = 500;
        double h             = 2. * radius;
        double alpha         = PI / 4;
        plane<dim> plane(
            {
                {0., 0.}
        },
            PI / 2);
        sphere<dim> sphere(
            {
                {0, h}
        },
            radius);
        particles.push_back(plane, property<dim>().deactivate());
        particles.push_back(sphere,
                            property<dim>()
                                .mass(1)
                                .moment_inertia(1. * radius * radius / 2.)
                                .force({
                                    {g * sin(alpha), -g * cos(alpha)}
        }));

        using problem_t    = ViscousFriction;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);
        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        double mu                                            = 0.5;
        params.default_contact_property.mu                   = mu;
        params.default_contact_property.fixed_point_tol      = 1e-3;
        params.default_contact_property.fixed_point_max_iter = 1000;
        params.default_contact_property.gamma                = 0;
        double gamma_min                                     = -1.4;
        params.default_contact_property.gamma_min            = gamma_min;
        params.default_contact_property.gamma_tol            = 1e-6;

        solver.run(dt, total_it);
        particles.f()(1)  = {-g * sin(alpha), g * cos(alpha)};
        double total_it_2 = 100;
        solver.run(dt, total_it_2);

        auto pos              = particles.pos();
        auto q                = particles.q();
        auto tmp              = analytical_solution_sphere_plane_viscous_friction(alpha,
                                                                     mu,
                                                                     dt * (total_it + total_it_2),
                                                                     radius,
                                                                     g,
                                                                     h,
                                                                     gamma_min,
                                                                     dt * (total_it));
        auto pos_analytical   = tmp.first;
        auto q_analytical     = quaternion(tmp.second);
        double error_pos      = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        double error_q        = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);
        auto v                = particles.v();
        auto omega            = particles.omega();
        tmp                   = analytical_solution_sphere_plane_velocity_viscous_friction(alpha,
                                                                         mu,
                                                                         dt * (total_it + total_it_2),
                                                                         radius,
                                                                         g,
                                                                         h,
                                                                         gamma_min,
                                                                         dt * (total_it));
        auto v_analytical     = tmp.first;
        auto omega_analytical = tmp.second;
        double error_v        = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);
        double error_omega    = std::norm(omega(1) - omega_analytical) / std::norm(omega_analytical);

        REQUIRE(error_pos <= doctest::Approx(1e-1));
        REQUIRE(error_q <= doctest::Approx(1e-1));
        REQUIRE(error_v <= doctest::Approx(1e-3));
        REQUIRE(error_omega <= doctest::Approx(1e-3));
    }

    /// TESTS 3 SPHERES

    TEST_CASE("3 Spheres NoFriction")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        sphere<dim> s1(
            {
                {-1.7, 1.3}
        },
            0.5);
        sphere<dim> s2(
            {
                {0.5, 1.7}
        },
            0.5);
        sphere<dim> s3(
            {
                {4.5, 1.3}
        },
            0.5);

        particles.push_back(s1,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {1.7, -1.3}
        }));

        particles.push_back(s2,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-0.5, -1.7}
        }));

        particles.push_back(s3,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-4.5, -1.3}
        }));

        double Tf = 6;
        double dt = 0.1;

        using problem_t    = NoFriction;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);

        fs::path path        = "test_gradient";
        std::string filename = "3spheres_nofriction";

        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        std::size_t total_it                  = Tf / dt;
        params.solver_params.output_frequency = total_it;
        params.solver_params.path             = path;
        params.solver_params.filename         = filename;
        solver.run(dt, total_it);

        CHECK(check_reference_file(path, filename, total_it, tol));
    }

    TEST_CASE("3 Spheres Viscous")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        sphere<dim> s1(
            {
                {-1.7, 1.3}
        },
            0.5);
        sphere<dim> s2(
            {
                {0.5, 1.7}
        },
            0.5);
        sphere<dim> s3(
            {
                {4.5, 1.3}
        },
            0.5);

        particles.push_back(s1,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {1.7, -1.3}
        }));

        particles.push_back(s2,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-0.5, -1.7}
        }));

        particles.push_back(s3,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-4.5, -1.3}
        }));

        double Tf = 7;
        double dt = 0.1;

        using problem_t    = Viscous;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);

        fs::path path        = "test_gradient";
        std::string filename = "3spheres_viscous";

        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        std::size_t total_it                  = Tf / dt;
        params.solver_params.output_frequency = total_it;
        params.solver_params.path             = path;
        params.solver_params.filename         = filename;
        solver.run(dt, total_it);

        CHECK(check_reference_file(path, filename, total_it, tol));
    }

    TEST_CASE("3 Spheres Friction Fixed Point")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        sphere<dim> s1(
            {
                {-1.7, 1.3}
        },
            0.5);
        sphere<dim> s2(
            {
                {0.5, 1.7}
        },
            0.5);
        sphere<dim> s3(
            {
                {4.5, 1.3}
        },
            0.5);

        particles.push_back(s1,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {1.7, -1.3}
        }));

        particles.push_back(s2,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-0.5, -1.7}
        }));

        particles.push_back(s3,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-4.5, -1.3}
        }));

        double Tf = 7;
        double dt = 0.1;

        using problem_t    = FrictionFixedPoint;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);

        fs::path path        = "test_gradient";
        std::string filename = "3spheres_friction_fp";

        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        // mu05
        params.default_contact_property.mu                   = 0.5;
        params.default_contact_property.fixed_point_tol      = 1e-6;
        params.default_contact_property.fixed_point_max_iter = 1000;

        std::size_t total_it                  = Tf / dt;
        params.solver_params.output_frequency = total_it;
        params.solver_params.path             = path;
        params.solver_params.filename         = filename;
        solver.run(dt, total_it);

        CHECK(check_reference_file(path, filename, total_it, tol));
    }

    TEST_CASE("3 Spheres Viscous Friction")
    {
        constexpr std::size_t dim = 2;

        scopi_container<dim> particles;

        sphere<dim> s1(
            {
                {-1.7, 1.3}
        },
            0.5);
        sphere<dim> s2(
            {
                {0.5, 1.7}
        },
            0.5);
        sphere<dim> s3(
            {
                {4.5, 1.3}
        },
            0.5);

        particles.push_back(s1,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {1.7, -1.3}
        }));

        particles.push_back(s2,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-0.5, -1.7}
        }));

        particles.push_back(s3,
                            property<dim>().mass(1).moment_inertia(1).force({
                                {-4.5, -1.3}
        }));

        double Tf = 7;
        double dt = 0.1;

        using problem_t    = ViscousFriction;
        using optim_solver = OptimGradient<apgd>;
        using vap_t        = vap_fpd;
        ScopiSolver<dim, problem_t, optim_solver, contact_kdtree, vap_t> solver(particles);

        fs::path path        = "test_gradient";
        std::string filename = "3spheres_viscous_friction";

        auto params = solver.get_params();

        params.optim_params.tolerance       = 1e-7;
        params.optim_params.max_ite         = 10000;
        params.optim_params.alpha           = 0.1;
        params.optim_params.dynamic_descent = true;

        params.default_contact_property.mu                   = 0.5;
        params.default_contact_property.fixed_point_tol      = 1e-3;
        params.default_contact_property.fixed_point_max_iter = 1000;
        params.default_contact_property.gamma                = 0;
        params.default_contact_property.gamma_min            = -1.4;
        params.default_contact_property.gamma_tol            = 1e-6;

        std::size_t total_it                  = Tf / dt;
        params.solver_params.output_frequency = total_it;
        params.solver_params.path             = path;
        params.solver_params.filename         = filename;
        solver.run(dt, total_it);

        CHECK(check_reference_file(path, filename, total_it, tol));
    }
}
