#include <iostream>

#include "scopi/soa.hpp"

struct particle
{
    double x, y, z;
    double radius;
};

SOA_DEFINE_TYPE(particle, x, y, z, radius);

int main()
{

    soa::vector<particle> v;
    v.resize(20);

    v.x[0] = 1.2;

    std::cout << v.x[0] << "\n";
    std::cout << v[0].x << "\n";
    std::cout << v.size() << "\n";

    auto print = [](const auto &particle) { std::cout << particle.x << "\n"; };
    auto print_ = [](const auto &x) { std::cout << x << "\n"; };
    std::for_each(v.rbegin(), v.rend(), print);
    std::cout << "\n";

    std::for_each(v.x.rbegin(), v.x.rend(), print_);
    std::cout << "\n";

    soa::vector<particle> v1;
    v1 = std::move(v);
    std::for_each(v1.begin(), v1.end(), print);
    std::cout << "\n";
    v1.push_back({1., 1., 1., 0});
    std::cout << v.size() << "\n";
    std::cout << v.capacity() << "\n";

    return 0;
}
