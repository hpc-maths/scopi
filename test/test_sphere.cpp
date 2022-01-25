#include <gtest/gtest.h>

#include <scopi/objects/types/sphere.hpp>
#include <scopi/container.hpp>
#include <scopi/objects/methods/closest_points.hpp>

namespace scopi
{
    TEST(sphere, distance)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(sphere, distance_1)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
        particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.dij, 0.2);
    }

}