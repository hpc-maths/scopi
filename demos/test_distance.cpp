#include <iostream>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/object/sphere.hpp>
#include <scopi/object/plan.hpp>
#include <scopi/object/neighbor.hpp>

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
    auto s3_pos = xt::eval(translation - 0.2*xt::view(rotation, xt::all(), 0));

    scopi::sphere<dim> s1({s1_pos}, 0.4);
    scopi::sphere<dim> s2({s2_pos}, 0.4);
    scopi::sphere<dim> s3({s3_pos}, 0.3);
    scopi::plan<dim> p1({translation}, theta);

    scopi::scopi_container<dim> particles;

    scopi::type::position<dim> dummy = {0, 0};

    particles.push_back(s1, {dummy}, {dummy}, {dummy});
    particles.push_back(s2, {dummy}, {dummy}, {dummy});
    particles.push_back(s3, {dummy}, {dummy}, {dummy});
    particles.push_back(p1, {dummy}, {dummy}, {dummy});

    std::vector<scopi::neighbor<dim>> contacts;
    double dmax = 0.05;

    for(std::size_t i = 0; i < particles.size() - 1; ++i)
    {
        for(std::size_t j = i + 1; j < particles.size(); ++j)
        {
            auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
            if (neigh.dij < dmax)
            {
                contacts.emplace_back(std::move(neigh));
            }
        }
    }
    std::cout << contacts.size() << std::endl;
}
