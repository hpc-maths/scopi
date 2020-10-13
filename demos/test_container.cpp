#include <iostream>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/shape/sphere.hpp>
#include <scopi/shape/globule.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    constexpr std::size_t size = 1000;
    scopi::sphere<dim> s1({1, 2}, 0.5);
    // scopi::sphere<dim> s2({5, 2}, 0.4);
    // scopi::globule<dim> g1({0, 3}, 0.1), g2({1, 3}, 0.2);
    scopi::scopi_container<dim> particles;

    particles.reserve(size);
    for(std::size_t i = 0; i < size; ++i)
    {
        particles.push_back(s1);
    }

    // particles.pos() *= 2;
    // particles.push_back(g1);
    // particles.push_back(g2);
    // particles.push_back(s1);
    // particles.push_back(s2);

    // std::cout << particles.pos() << "\n\n";

    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        // particles[i]->print();
        particles[i]->pos().fill(2);
        // particles[i]->pos() += 2;
        // std::cout << particles[i]->pos() << "\n\n";
    }
    // std::cout << particles.pos() << "\n\n";

}