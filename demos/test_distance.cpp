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

    scopi::type::position<dim> translation = {3, 1};

    auto s1_pos = xt::eval(translation + 0.5*xt::view(rotation, xt::all(), 0));
    auto s2_pos = xt::eval(translation - 0.5*xt::view(rotation, xt::all(), 0));
    scopi::sphere<dim> s1({s1_pos}, 0.4);
    scopi::sphere<dim> s2({s2_pos}, 0.4);

    scopi::type::position<dim> pos1 = {1, -2};
    scopi::type::position<dim> radius1 = {1, 1};
    scopi::type::position<dim-1> squareness1 = {0.5};
    // scopi::type::quaternion quat1 = { std::cos(0.5*theta), 0, 0, std::sin(0.5*theta) };
    scopi::type::quaternion quat1 = { -0.66693806,  0, 0, 0.74511316 };
    scopi::superellipsoid<dim> se1({pos1}, {quat1}, {radius1}, {squareness1});
    se1.print();
    scopi::type::position<dim> pos2 = {12, 4};
    scopi::type::position<dim> radius2 = {2, 3};
    scopi::type::position<dim-1> squareness2 = {1.};
    scopi::type::quaternion quat2 = { 1,  0, 0, 0 };
    scopi::superellipsoid<dim> se2({pos2}, {quat2}, {radius2}, {squareness2});
    se2.print();
    // std::cout << "se1.point(0.43)   = " << se1.point(0.43) <<   " se2.point(0.65)   = " << se2.point(0.65) << std::endl;
    // std::cout << "se1.normal(0.43)  = " << se1.normal(0.43) <<  " se2.normal(0.65)  = " << se2.normal(0.65) << std::endl;
    // std::cout << "se1.tangent(0.43) = " << se1.tangent(0.43) << " se2.tangent(0.65) = " << se2.tangent(0.65) << std::endl;


    scopi::plan<dim> p1({translation}, theta);

    scopi::scopi_container<dim> particles;

    scopi::type::position<dim> dummy = {0, 0};

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
