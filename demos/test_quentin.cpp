#include <cstdlib>
#include <vector>

#include <scopi/container.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>

#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xnorm.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/matrix/velocities.hpp>
#include <scopi/solvers/OptimGradient.hpp>
#include <scopi/solvers/apgd.hpp>

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

// int nofriction_two_spheres(int argc, char** argv)
// {
//     CLI::App app("Two spheres with no friction");

//     constexpr std::size_t dim = 2;

//     scopi::scopi_container<dim> particles;

//     scopi::sphere<dim> s1(
//         {
//             {1.5, 1.6}
//     },
//         0.5);

//     scopi::sphere<dim> s2(
//         {
//             {3.5, 1.4}
//     },
//         0.5);

//     particles.push_back(s1,
//                         scopi::property<dim>()
//                             .mass(1)
//                             .velocity({
//                                 {1, 0}
//     })
//                             .moment_inertia(1));
//     particles.push_back(s2,
//                         scopi::property<dim>()
//                             .mass(1)
//                             .velocity({
//                                 {-1, 0}
//     })
//                             .moment_inertia(1));

//     double Tf = 14;
//     double dt = 0.05;

//     using problem_t    = scopi::NoFriction;
//     using optim_solver = scopi::OptimGradient<scopi::apgd>;
//     using contact_t    = scopi::contact_kdtree;
//     using vap_t        = scopi::vap_fpd;
//     scopi::ScopiSolver<dim, problem_t, optim_solver, contact_t, vap_t> solver(particles, dt);
//     solver.init_options(app);
//     CLI11_PARSE(app, argc, argv);
//     solver.run(Tf / dt);
//     return 0;
// }

// int nofriction_plan_sphere(int argc, char** argv)
// {
//     CLI::App app("Plan/sphere with no friction");

//     constexpr std::size_t dim = 2;

//     scopi::scopi_container<dim> particles;

//     double PI = xt::numeric_constants<double>::PI;
//     scopi::plan<dim> plan(
//         {
//             {0., 0.}
//     },
//         PI / 2 - PI / 4);
//     scopi::sphere<dim> sphere(
//         {
//             {0, 1.5}
//     },
//         0.5);
//     particles.push_back(plan, scopi::property<dim>().deactivate());
//     particles.push_back(sphere,
//                         scopi::property<dim>().mass(1).moment_inertia(1).velocity({
//                             {0, -1}
//     }));

//     double Tf = 2.5;
//     double dt = 0.05;

//     using problem_t    = scopi::NoFriction;
//     using optim_solver = scopi::OptimGradient<scopi::apgd>;
//     using contact_t    = scopi::contact_kdtree;
//     using vap_t        = scopi::vap_fpd;
//     scopi::ScopiSolver<dim, problem_t, optim_solver, contact_t, vap_t> solver(particles, dt);
//     solver.init_options(app);
//     CLI11_PARSE(app, argc, argv);
//     solver.run(Tf / dt);
//     return 0;
// }

// int viscous_two_spheres(int argc, char** argv)
// {
//     CLI::App app("Two viscous spheres");

//     constexpr std::size_t dim = 2;

//     scopi::scopi_container<dim> particles;

//     scopi::sphere<dim> s1(
//         {
//             {1.5, 1.6}
//     },
//         0.5);

//     scopi::sphere<dim> s2(
//         {
//             {3.5, 1.4}
//     },
//         0.5);

//     particles.push_back(s1,
//                         scopi::property<dim>()
//                             .mass(1)
//                             .force({
//                                 {1, 0}
//     })
//                             .moment_inertia(1));
//     particles.push_back(s2,
//                         scopi::property<dim>()
//                             .mass(1)
//                             .force({
//                                 {-1, 0}
//     })
//                             .moment_inertia(1));

//     double Tf = 5;
//     double dt = 0.05;

//     using problem_t    = scopi::Viscous;
//     using optim_solver = scopi::OptimGradient<scopi::apgd>;
//     using contact_t    = scopi::contact_kdtree;
//     using vap_t        = scopi::vap_fpd;
//     scopi::ScopiSolver<dim, problem_t, optim_solver, contact_t, vap_t> solver(particles, dt);
//     solver.init_options(app);
//     CLI11_PARSE(app, argc, argv);
//     solver.run(Tf / dt);
//     return 0;
// }

int viscous_plan_sphere(int argc, char** argv)
{
    CLI::App app("viscous plan/sphere");

    constexpr std::size_t dim = 2;

    scopi::scopi_container<dim> particles;

    const double PI = xt::numeric_constants<double>::PI;
    scopi::plan<dim> plan(
        {
            {0., 0.}
    },
        PI / 2);
    scopi::sphere<dim> sphere(
        {
            {0, 1.}
    },
        0.5);
    particles.push_back(plan, scopi::property<dim>().deactivate());
    particles.push_back(sphere,
                        scopi::property<dim>().mass(1).moment_inertia(1).force({
                            {0, -1}
    }));

    double Tf = 2.5;
    double dt = 0.05;

    using problem_t    = scopi::Viscous;
    using optim_solver = scopi::OptimGradient<scopi::apgd>;
    using contact_t    = scopi::contact_kdtree;
    using vap_t        = scopi::vap_fpd;
    scopi::ScopiSolver<dim, problem_t, optim_solver, contact_t, vap_t> solver(particles, dt);
    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(Tf / dt);

    Tf               = 5.;
    particles.f()(1) = {0, 1};
    solver.run(Tf / dt);

    return 0;
}

int friction_plan_sphere(int argc, char** argv)
{
    CLI::App app("friction plan/sphere");

    constexpr std::size_t dim = 2;

    scopi::scopi_container<dim> particles;

    double PI = xt::numeric_constants<double>::PI;
    scopi::plan<dim> plan(
        {
            {0., 0.}
    },
        PI / 2 - PI / 6);

    double dd = 1;
    double rr = 2;
    scopi::sphere<dim> sphere(
        {
            {0, (rr + dd) / std::cos(PI / 6)}
    },
        rr);
    particles.push_back(plan, scopi::property<dim>().deactivate());
    particles.push_back(sphere,
                        scopi::property<dim>()
                            .mass(1)
                            .moment_inertia(0.5 * rr * rr)
                            .force({
                                {0, -1}
    }));

    double Tf = 10;
    double dt = 0.1;

    using problem_t    = scopi::Friction;
    using optim_solver = scopi::OptimGradient<scopi::apgd>;
    using contact_t    = scopi::contact_kdtree;
    using vap_t        = scopi::vap_fpd;
    scopi::ScopiSolver<dim, problem_t, optim_solver, contact_t, vap_t> solver(particles, dt);
    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(Tf / dt);

    return 0;
}

int main(int argc, char** argv)
{
    // static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    // plog::init(plog::info, &consoleAppender);

    // std::setprecision(15);
    // xt::print_options::set_precision(15);

    // nofriction_two_spheres(argc, argv);
    // nofriction_plan_sphere(argc, argv);
    // viscous_two_spheres(argc, argv);
    // viscous_plan_sphere(argc, argv);
    friction_plan_sphere(argc, argv);

    return 0;
}
