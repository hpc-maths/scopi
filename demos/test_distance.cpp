#include <iostream>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/object/sphere.hpp>
#include <scopi/object/superellipsoid.hpp>
#include <scopi/object/plan.hpp>

#include <scopi/functors.hpp>
#include <scopi/types.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double theta = 4*std::atan(1.)/4;

    scopi::type::rotation<dim> rotation{{{std::cos(theta), -std::sin(theta)},
                                         {std::sin(theta), std::cos(theta)}} };

    scopi::type::position<dim> translation = {0, 0};

    auto s1_pos = xt::eval(translation + 0.5*xt::view(rotation, xt::all(), 0));
    auto s2_pos = xt::eval(translation - 0.5*xt::view(rotation, xt::all(), 0));
    scopi::sphere<dim> s1({s1_pos}, 0.4);
    scopi::sphere<dim> s2({s2_pos}, 0.4);

    scopi::type::position<dim> radius = {1, 2};
    scopi::type::position<dim-1> squareness = {2};
    scopi::type::quaternion quat1 = { std::cos(0.5*theta), 0, 0, std::sin(0.5*theta) };
    scopi::type::quaternion quat2 = { std::cos(0.5*theta/2), 0, 0, std::sin(0.5*theta/2) };
    auto se1_pos = xt::eval(translation + 5*xt::view(rotation, xt::all(), 0));
    auto se2_pos = xt::eval(translation - 5*xt::view(rotation, xt::all(), 0));
    scopi::superellipsoid<dim> se1({se1_pos}, {quat1}, {radius}, {squareness});
    scopi::superellipsoid<dim> se2({se2_pos}, {quat2}, {radius}, {squareness});

    scopi::plan<dim> p1({translation}, theta);

    scopi::scopi_container<dim> particles;

    scopi::type::position<dim> dummy = {0, 0};

    particles.push_back(s1, {dummy}, {dummy}, {dummy});
    particles.push_back(s2, {dummy}, {dummy}, {dummy});
    particles.push_back(p1, {dummy}, {dummy}, {dummy});
    particles.push_back(se1, {dummy}, {dummy}, {dummy});
    particles.push_back(se2, {dummy}, {dummy}, {dummy});

    for(std::size_t i = 0; i < particles.size() - 1; ++i)
    {
        for(std::size_t j = i + 1; j < particles.size(); ++j)
        {
            std::cout << scopi::distance_dispatcher<dim>::dispatch(*particles[i], *particles[j]) << std::endl;
            std::cout << scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]) << std::endl;
        }
    }

    std::cout << "particles.pos() = \n" << particles.pos() << "\n\n";
}
