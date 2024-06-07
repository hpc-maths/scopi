#include <scopi/container.hpp>
#include <scopi/objects/types/worm.hpp>
#include <scopi/scopi.hpp>
#include <scopi/solver.hpp>

int main(int argc, char** argv)
{
    scopi::initialize("Two worms simulation");

    constexpr std::size_t dim = 2;
    double dt                 = .005;
    std::size_t total_it      = 1000;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::worm<dim> w1(
        {
            {1.,  0.7},
            {3.,  0.7},
            {5.,  0.7},
            {7.,  0.7},
            {9.,  0.7},
            {11., 0.7}
    },
        {{scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)}},
        1.,
        6);
    scopi::worm<dim> w2(
        {
            {-1.,  -0.7},
            {-3.,  -0.7},
            {-5.,  -0.7},
            {-7.,  -0.7},
            {-9.,  -0.7},
            {-11., -0.7}
    },
        {{scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)},
         {scopi::quaternion(0.)}},
        1.,
        6);
    particles.push_back(w1, prop.desired_velocity({-1., 0.}));
    particles.push_back(w2, prop.desired_velocity({1., 0.}));

    scopi::ScopiSolver<dim> solver(particles);
    SCOPI_PARSE(argc, argv);

    solver.run(dt, total_it);

    return 0;
}
