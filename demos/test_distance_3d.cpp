#include <iostream>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/object/sphere.hpp>
#include <scopi/object/plan.hpp>

#include <scopi/functors.hpp>
#include <scopi/types.hpp>

int main()
{
    constexpr std::size_t dim = 3;
    double theta = 4*std::atan(1.)/4;

    scopi::type::rotation<dim> rotation{{{std::cos(theta), -std::sin(theta), 0},
                                         {std::sin(theta),  std::cos(theta), 0},
                                         {              0,                0, 1}}};

    scopi::type::position<dim> translation = {0, 0, 0};

    auto s1_pos = xt::eval(translation + 0.5*xt::view(rotation, xt::all(), 0));
    auto s2_pos = xt::eval(translation - 0.5*xt::view(rotation, xt::all(), 0));

    scopi::sphere<dim> s1({s1_pos}, 0.4);
    scopi::sphere<dim> s2({s2_pos}, 0.4);

    scopi::type::position<dim> pos1 = {0, 0, 0};
    scopi::type::position<dim> radius1 = {2, 1, 1};
    scopi::type::position<dim-1> squareness1 = {0.8, 0.9};
    // scopi::type::quaternion quat1 = { std::cos(0.5*theta), 0, 0, std::sin(0.5*theta) };
    scopi::type::quaternion quat1 = { 0.46773184, 0., -0.62499077, 0.62499077 };
    scopi::superellipsoid<dim> se1({pos1}, {quat1}, {radius1}, {squareness1});
    se1.print();
    scopi::type::position<dim> pos2 = {5, 4, 4};
    scopi::type::position<dim> radius2 = {1, 1, 1};
    scopi::type::position<dim-1> squareness2 = {0.9, 0.4};
    scopi::type::quaternion quat2 = { 0.75390225, 0.37931139, 0.37931139, 0.37931139 };
    scopi::superellipsoid<dim> se2({pos2}, {quat2}, {radius2}, {squareness2});
    se2.print();
    // std::cout << "se1.point(0.43,0.65)   = " << se1.point(0.43,0.65) <<   " se2.point(0.65,0.43)   = " << se2.point(0.65,0.43) << std::endl;
    // std::cout << "se1.normal(0.43,0.65)  = " << se1.normal(0.43,0.65) <<  " se2.normal(0.65,0.43)  = " << se2.normal(0.65,0.43) << std::endl;
    // auto tgts1 = se1.tangents(0.43,0.65);
    // auto tgts2 = se2.tangents(0.65,0.43);
    // std::cout << "se1.tangents(0.43,0.65) = " << tgts1.first << tgts1.second << " se2.tangents(0.65,0.43) = " << tgts2.first << tgts2.second << std::endl;

    scopi::plan<dim> p1({translation}, theta);

    scopi::scopi_container<dim> particles;

    scopi::type::position<dim> dummy = {0, 0, 0};

    // particles.push_back(s1, {dummy}, {dummy}, {dummy});
    // particles.push_back(s2, {dummy}, {dummy}, {dummy});
    // particles.push_back(p1, {dummy}, {dummy}, {dummy});
    particles.push_back(se1, {dummy}, {dummy}, {dummy});
    particles.push_back(se2, {dummy}, {dummy}, {dummy});

    for(std::size_t i = 0; i < particles.size() - 1; ++i)
    {
        for(std::size_t j = i + 1; j < particles.size(); ++j)
        {
            // std::cout << scopi::distance_dispatcher<dim>::dispatch(*particles[i], *particles[j]) << std::endl;
            std::cout << scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]) << std::endl;
        }
    }
    // std::cout << "particles.pos() = \n" << particles.pos() << "\n\n";
}
