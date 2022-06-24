#include <cstddef>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/worm.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/problems/ViscousGlobule.hpp>
#include <scopi/contact/contact_brute_force.hpp>

int main()
{
    plog::init(plog::error, "two_worms.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::worm<dim> g1({{1., 0.7}, {3., 0.7}, {5., 0.7}, {7., 0.7}, {9., 0.7}, {11., 0.7}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            1.);
    scopi::worm<dim> g2({{-1., -0.7}, {-3., -0.7}, {-5., -0.7}, {-7., -0.7}, {-9., -0.7}, {-11., -0.7}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            1.);
    particles.push_back(g1, prop.desired_velocity({-1., 0.}));
    particles.push_back(g2, prop.desired_velocity({1., 0.}));

    scopi::OptimParams<scopi::OptimMosek<scopi::ViscousGlobule>> params;
    params.m_change_default_tol_mosek = false;
    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::ViscousGlobule>, scopi::contact_kdtree> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
