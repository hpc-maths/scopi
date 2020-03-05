#include "sphere.hpp"

int main()
{
    scopi::position p;
    scopi::quaternion q;
    scopi::sphere s1{p, 1, q}, s2{p, 2, q};

    auto o = scopi::make_object<scopi::globule>(s1, s2);

    return 0;
}