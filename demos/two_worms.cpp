#include <cstddef>
#include <CLI/CLI.hpp>

#include <xtensor/xmath.hpp>
#include <scopi/objects/types/worm.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/contact/contact_brute_force.hpp>

int main(int argc, char **argv)
{
    plog::init(plog::error, "two_worms.log");

    CLI::App app("two spheres with periodic boundary conditions");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::worm<dim> w1({{1., 0.7}, {3., 0.7}, {5., 0.7}, {7., 0.7}, {9., 0.7}, {11., 0.7}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            1., 6);
    scopi::worm<dim> w2({{-1., -0.7}, {-3., -0.7}, {-5., -0.7}, {-7., -0.7}, {-9., -0.7}, {-11., -0.7}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            1., 6);
    particles.push_back(w1, prop.desired_velocity({-1., 0.}));
    particles.push_back(w2, prop.desired_velocity({1., 0.}));


//     scopi::Params<scopi::OptimMosek<scopi::DryWithoutFriction>, scopi::contact_kdtree, scopi::vap_fixed> params;
//     params.optim_params.change_default_tol_mosek = false;
//     scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::DryWithoutFriction>, scopi::contact_kdtree> solver(particles, dt, params);
//     solver.run(total_it);

    scopi::ScopiSolver<dim,
                       scopi::OptimUzawaMatrixFreeOmp<scopi::DryWithoutFriction>,
                       scopi::contact_brute_force> solver(particles, dt);
    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(total_it);

    return 0;
}
