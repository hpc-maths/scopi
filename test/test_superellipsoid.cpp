#include <gtest/gtest.h>
#include "utils.hpp"

#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>
#include <scopi/solvers/mosek.hpp>

namespace scopi
{
    // pos
    TEST(superellipsoid, pos_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d_const)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_index)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(superellipsoid, pos_2d_index_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index_const)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(superellipsoid, pos_2d_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST(superellipsoid, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    // q
    TEST(superellipsoid, q)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(superellipsoid, q_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(superellipsoid, q_index)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(superellipsoid, q_index_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(superellipsoid, q_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(superellipsoid, q_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST(superellipsoid, q_index_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    TEST(superellipsoid, q_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    // radius
    TEST(superellipsoid, radius_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1}});

        auto radius = s.radius();

        EXPECT_EQ(radius(0), 0.1);
        EXPECT_EQ(radius(1), 0.2);
    }

    TEST(superellipsoid, radius_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto radius = s.radius();

        EXPECT_EQ(radius(0), 0.1);
        EXPECT_EQ(radius(1), 0.2);
        EXPECT_EQ(radius(2), 0.3);
    }

    //squareness
    TEST(superellipsoid, squareness_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(0)}, {{0.1, 0.2}}, {{1.5}});

        auto squareness = s.squareness();

        EXPECT_EQ(squareness(0), 1.5);
    }

    TEST(superellipsoid, squareness_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}});

        auto squareness = s.squareness();

        EXPECT_EQ(squareness(0), 0.5);
        EXPECT_EQ(squareness(1), 1.5);
    }

    // rotation
    TEST(superellipsoid, rotation_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {quaternion(PI/3)}, {{0.1, 0.2}}, {{1}});

        auto rotation_matrix = s.rotation();

        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
    }

    TEST(superellipsoid, rotation_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {quaternion(PI/3    )}, {{0.1, 0.2, 0.3}}, {{1, 1}});

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
    TEST(superellipsoid, point_x_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto point = s.point(0.);

        EXPECT_DOUBLE_EQ(point(0), 0.1);
        EXPECT_DOUBLE_EQ(point(1), 0.);
    }

    TEST(superellipsoid, point_y_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto point = s.point(PI/2.);

        EXPECT_NEAR(point(0), 0., 1e-17); // TODO point(0) = 6e-18, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(point(1), 0.2);
    }

    TEST(superellipsoid, point_x_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto point = s.point(0., 0.);

        EXPECT_DOUBLE_EQ(point(0), 0.1);
        EXPECT_DOUBLE_EQ(point(1), 0.);
        EXPECT_DOUBLE_EQ(point(2), 0.);
    }

    TEST(superellipsoid, point_y_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto point = s.point(0., PI/2.);

        EXPECT_NEAR(point(0), 0., 1e-17); // TODO point(0) = 6e-18, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(point(1), 0.2);
        EXPECT_DOUBLE_EQ(point(2), 0.);
    }

    TEST(superellipsoid, point_z_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto point = s.point(PI/2., 0.);

        EXPECT_NEAR(point(0), 0., 1e-17); // TODO point(0) = 6e-18, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(point(1), 0.);
        EXPECT_DOUBLE_EQ(point(2), 0.3);
    }

    // normal
    TEST(superellipsoid, normal_x_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto normal = s.normal(0.);

        EXPECT_DOUBLE_EQ(normal(0), 1.);
        EXPECT_DOUBLE_EQ(normal(1), 0.);
    }

    TEST(superellipsoid, normal_y_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto normal = s.normal(PI/2.);

        EXPECT_NEAR(normal(0), 0., 5e-16); // TODO point(0) = 1e-16, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(normal(1), 1.);
    }

    TEST(superellipsoid, normal_x_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto normal = s.normal(0., 0.);

        EXPECT_DOUBLE_EQ(normal(0), 1.);
        EXPECT_DOUBLE_EQ(normal(1), 0.);
        EXPECT_DOUBLE_EQ(normal(2), 0.);
    }

    TEST(superellipsoid, normal_y_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto normal = s.normal(0., PI/2.);

        EXPECT_NEAR(normal(0), 0., 5e-16); // TODO point(0) = 1e-16, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(normal(1), 1.);
        EXPECT_DOUBLE_EQ(normal(2), 0.);
    }

    TEST(superellipsoid, normal_z_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto normal = s.normal(PI/2., 0.);

        EXPECT_NEAR(normal(0), 0., 5e-16); // TODO point(0) = 1e-16, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(normal(1), 0.);
        EXPECT_DOUBLE_EQ(normal(2), 1.);
    }

    // tangent
    TEST(superellipsoid, tangent_x_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto tangent = s.tangent(0.);

        EXPECT_DOUBLE_EQ(tangent(0), 0.);
        EXPECT_DOUBLE_EQ(tangent(1), 1.);
    }

    TEST(superellipsoid, tangent_y_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto tangent = s.tangent(PI/2.);

        EXPECT_DOUBLE_EQ(tangent(0), -1.);
        EXPECT_NEAR(tangent(1), 0., 5e-16); // TODO point(0) = 1e-16, EXPECT_DOUBLE_EQ fails
    }

    TEST(superellipsoid, tangent_x_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto tangent = s.tangents(0., 0.);

        EXPECT_DOUBLE_EQ(tangent.first(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.first(1), 1.);
        EXPECT_DOUBLE_EQ(tangent.first(2), 0.);

        EXPECT_DOUBLE_EQ(tangent.second(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(1), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(2), 1.);
    }

    TEST(superellipsoid, tangent_y_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto tangent = s.tangents(0., PI/2.);

        EXPECT_DOUBLE_EQ(tangent.first(0), -1.);
        EXPECT_NEAR(tangent.first(1), 0., 5e-16); // TODO point(0) = 1e-16, EXPECT_DOUBLE_EQ fails
        EXPECT_DOUBLE_EQ(tangent.first(2), 0.);

        EXPECT_DOUBLE_EQ(tangent.second(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(1), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(2), 1.);
    }

    TEST(superellipsoid, tangent_z_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto tangent = s.tangents(PI/2., 0.);

        EXPECT_DOUBLE_EQ(tangent.first(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.first(1), 1.);
        EXPECT_DOUBLE_EQ(tangent.first(2), 0.);

        EXPECT_DOUBLE_EQ(tangent.second(0), -1.);
        EXPECT_DOUBLE_EQ(tangent.second(1), 0.);
        EXPECT_NEAR(tangent.second(2), 0., 5e-16); // TODO point(0) = 1e-16, EXPECT_DOUBLE_EQ fails
    }

    // two ellipsoids
    class TestTwoEllipsoidsSymmetrical  : public ::testing::Test {
        protected:
            void SetUp() override {
                superellipsoid<2> s1({{-0.2, 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05}}, {{1}});
                superellipsoid<2> s2({{0.2, 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05}}, {{1}});
                m_particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
                m_particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});
            }

            double m_dt = .005;
            std::size_t m_total_it = 200;
            scopi_container<2> m_particles;
            std::size_t m_active_ptr = 0; // without obstacles
    };

    TEST_F(TestTwoEllipsoidsSymmetrical, two_ellipsoids_symmetrical)
    {
        // TODO set the optimization solver (Mosek, Uzawa, ...) here and duplicate this test for all solver 
        constexpr std::size_t dim = 2;
        ScopiSolver<dim> solver(m_particles, m_dt, m_active_ptr);
        solver.solve(m_total_it);

        std::string filenameRef;
        if(solver.getOptimSolverName() == "OptimMosek")
            filenameRef = "../test/two_ellipsoids_symmetrical_mosek.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMkl")
            filenameRef = "../test/two_ellipsoids_symmetrical_uzawaMkl.json"; 
        else if(solver.getOptimSolverName() == "OptimScs")
            filenameRef = "../test/two_ellipsoids_symmetrical_scs.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeTbb")
            filenameRef = "../test/two_ellipsoids_symmetrical_uzawaMatrixFreeTbb.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeOmp")
            filenameRef = "../test/two_ellipsoids_symmetrical_uzawaMatrixFreeOmp.json"; 

        EXPECT_PRED2(diffFile, "./Results/scopi_objects_0199.json", filenameRef);
    }


}
