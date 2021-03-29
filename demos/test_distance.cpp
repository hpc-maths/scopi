#include <iostream>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/object/sphere.hpp>
#include <scopi/object/plan.hpp>

#include <scopi/functors.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double theta = 4*std::atan(1.)/4;
    std::array<std::array<double, dim>, dim> rotation{ {{std::cos(theta), -std::sin(theta)},
                                                        {std::sin(theta), std::cos(theta)}} };
    std::array<std::array<double, dim>, dim> rotation_inv{ {{std::cos(-theta), -std::sin(-theta)},
                                                            {std::sin(-theta), std::cos(-theta)}} };
    std::array<double, dim> translation{{0, 0}};

    std::array<double, dim> s1_pos;
    std::array<double, dim> s2_pos;

    for (std::size_t d=0; d<dim; ++d)
    {
        s1_pos[d] = translation[d] + rotation[d][0] * 0.5;
        s2_pos[d] = translation[d] - rotation[d][0] * 0.5;
    }
    scopi::sphere<dim> s1({s1_pos}, 0.4);
    scopi::sphere<dim> s2({s2_pos}, 0.4);
    // scopi::plan<dim> p1({{0.5, 0}});

    scopi::plan<dim> p1({translation}, {rotation_inv});

    scopi::scopi_container<dim> particles;

    std::array<double, dim> dummy{{0, 0}};

    particles.push_back(s1, {dummy}, {dummy}, {dummy});
    particles.push_back(s2, {dummy}, {dummy}, {dummy});
    particles.push_back(p1, {dummy}, {dummy}, {dummy});

    for(std::size_t i = 0; i < particles.size() - 1; ++i)
    {
        for(std::size_t j = i + 1; j < particles.size(); ++j)
        {
            std::cout << scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]) << std::endl;
        }
    }

    // // std::cout << "particles.pos() = \n" << particles.pos() << "\n\n";
}
