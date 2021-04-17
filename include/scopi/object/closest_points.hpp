#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>

#include "sphere.hpp"
#include "superellipsoid.hpp"
#include "globule.hpp"
#include "plan.hpp"

namespace scopi
{
    // SPHERE - SPHERE
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s1, const sphere<dim, false>& s2)
    {
        std::cout << "closest_points : SPHERE - SPHERE" << std::endl;
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto s1_pos = xt::view(s1.pos(), 0);
        auto s2_pos = xt::view(s2.pos(), 0);
        auto s1_to_s2 = s2_pos - s1_pos;
        xt::view(pts, 0) = s1_pos + s1.radius()*s1_to_s2;
        xt::view(pts, 1) = s2_pos - s2.radius()*s1_to_s2;
        return pts;
    }

    // SUPERELLIPSOID - SUPERELLIPSOID
    template<std::size_t dim>
    auto closest_points(const superellipsoid<dim, false>& s1, const superellipsoid<dim, false>& s2)
    {
        std::cout << "closest_points : SUPERELLIPSOID - SUPERELLIPSOID" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // PLAN - PLAN
    template<std::size_t dim>
    auto closest_points(const plan<dim, false>, const plan<dim, false>)
    {
        std::cout << "closest_points : PLAN - PLAN" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // GLOBULE - GLOBULE
    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const globule<dim, false>)
    {
        std::cout << "closest_points : GLOBULE - GLOBULE" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SPHERE - GLOBULE
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>, const globule<dim, false>)
    {
        std::cout << "closest_points : SPHERE - GLOBULE" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SPHERE - PLAN
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s, const plan<dim, false>& p)
    {
        std::cout << "closest_points : SPHERE - PLAN" << std::endl;
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto s_pos = s.pos(0);
        auto p_pos = p.pos(0);

        auto normal = p.normal();

        // plan2sphs.n
        auto plan_to_sphere = xt::eval(xt::linalg::dot(s_pos - p_pos, normal));
        auto sign = xt::sign(plan_to_sphere);

        xt::view(pts, 0) = s_pos - sign*s.radius()*normal;
        xt::view(pts, 1) = s_pos - plan_to_sphere*normal;
        return pts;
    }

    // SPHERE - SUPERELLIPSOID
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>, const superellipsoid<dim, false>)
    {
        std::cout << "closest_points : SPHERE - SUPERELLIPSOID" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SUPERELLIPSOID - PLAN
    template<std::size_t dim>
    auto closest_points(const superellipsoid<dim, false>, const plan<dim, false>)
    {
        std::cout << "closest_points : SUPERELLIPSOID - PLAN" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SUPERELLIPSOID - GLOBULE
    template<std::size_t dim>
    auto closest_points(const superellipsoid<dim, false>, const globule<dim, false>)
    {
        std::cout << "closest_points : SUPERELLIPSOID - GLOBULE" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // GLOBULE - PLAN
    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const plan<dim, false>)
    {
        std::cout << "closest_points : GLOBULE - PLAN" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }
}
