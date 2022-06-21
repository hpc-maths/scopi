#include <cstddef>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/globule.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/problems/ViscousGlobule.hpp>

int main()
{
    plog::init(plog::error, "two_globules.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 2000;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::globule<dim> g1({{2., 0.7}, {4., 0.7}, {6., 0.7}, {8., 0.7}, {10., 0.7}, {12., 0.7}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            1.);
    scopi::globule<dim> g2({{-2., -0.7}, {-4., -0.7}, {-6., -0.7}, {-8., -0.7}, {-10., -0.7}, {-12., -0.7}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            1.);
    particles.push_back(g1, prop.desired_velocity({-1., 0.}));
    particles.push_back(g2, prop.desired_velocity({1., 0.}));

    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::ViscousGlobule>> solver(particles, dt);
    solver.solve(total_it);

    return 0;
}
