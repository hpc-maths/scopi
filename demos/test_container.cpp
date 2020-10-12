#include <iostream>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/shape/sphere.hpp>
#include <scopi/shape/globule.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    scopi::sphere<dim> s1({1, 2}, 0.5), s2({1, 2}, 0.5);
    scopi::globule<dim> g1({0, 3}, 0.1);
    scopi::scopi_container<dim> particles;

    particles.push_back(s1);
    particles.push_back(s2);

    std::cout << particles.pos() << "\n";
    particles[0]->print();

    g1.print();
    for(auto&p: g1.pos())
    {
        std::cout << xt::adapt(p) << "\n";
    }
}