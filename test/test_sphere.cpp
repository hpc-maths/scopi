#include <gtest/gtest.h>

#include <scopi/objects/types/sphere.hpp>
#include <scopi/container.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

namespace scopi
{
    // pos
    TEST(sphere, pos_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(sphere, pos_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(sphere, pos_3d_const)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_index)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_index_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_const)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(sphere, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(sphere, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    // q
    TEST(sphere, q)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(sphere, q_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(sphere, q_index)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(sphere, q_index_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(sphere, q_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(sphere, q_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST(sphere, q_index_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    TEST(sphere, q_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    // radius
    TEST(sphere, radius)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.radius(), 0.1);
    }

    // rotation
    TEST(sphere, rotation_2d)
    {
        constexpr std::size_t dim = 2;
        double PI = xt::numeric_constants<double>::PI;
        sphere<dim> s({{-0.2, 0.0}}, {quaternion(PI/3)}, 0.1);

        auto rotation_matrix = s.rotation();

        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
    }

    TEST(sphere, rotation_3d)
    {
        constexpr std::size_t dim = 3;
        double PI = xt::numeric_constants<double>::PI;
        sphere<dim> s({{-0.2, 0.0, 0.0}}, {quaternion(PI/3)}, 0.1);

        auto rotation_matrix = s.rotation();

        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 2), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 2), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(2, 0), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(2, 1), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(2, 2), 1.);
    }

    // point
    TEST(sphere, point_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);

        auto point = s.point(0.);

        EXPECT_EQ(point(0), 0.1);
        EXPECT_EQ(point(1), 0.);
    }

    TEST(sphere, point_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);

        auto point = s.point(0., 0.);

        EXPECT_EQ(point(0), 0.1);
        EXPECT_EQ(point(1), 0.);
        EXPECT_EQ(point(2), 0.);
    }

    // normal
    TEST(sphere, normal_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);

        auto normal = s.normal(0.);

        EXPECT_EQ(normal(0), 1.);
        EXPECT_EQ(normal(1), 0.);
    }

    TEST(sphere, normal_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);

        auto normal = s.normal(0., 0.);

        EXPECT_EQ(normal(0), 1.);
        EXPECT_EQ(normal(1), 0.);
        EXPECT_EQ(normal(2), 0.);
    }

    // two_spheres
    TEST(sphere, two_spheres)
    {
        std::ifstream file1("./Results/scopi_objects_0999.json");
        std::ifstream file2("./Results/scopi_objects_0999.json");
        if(file1 && file2)
        {
            nlohmann::json j1 = nlohmann::json::parse(file1);
            nlohmann::json j2 = nlohmann::json::parse(file2);
            nlohmann::json difference = nlohmann::json::diff(j1, j2);
            EXPECT_EQ(difference.empty(), true);
        }
        else
        {
            FAIL();
        }

    }

}
