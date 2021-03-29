#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>

#include "sphere.hpp"
#include "globule.hpp"
#include "plan.hpp"

namespace scopi
{
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s1, const sphere<dim, false>& s2)
    {
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto s1_pos = xt::view(s1.pos(), 0);
        auto s2_pos = xt::view(s2.pos(), 0);
        auto s1_to_s2 = s2_pos - s1_pos;
        xt::view(pts, 0) = s1_pos + s1.radius()*s1_to_s2;
        xt::view(pts, 1) = s2_pos - s2.radius()*s1_to_s2;
        return pts;
    }

    template<std::size_t dim>
    auto closest_points(const plan<dim, false>, const plan<dim, false>)
    {
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const globule<dim, false>)
    {
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>, const globule<dim, false>)
    {
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s, const plan<dim, false>& p)
    {
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        // nref = (1,0,0) => n = R nref
        auto s_pos = s.pos(0);
        auto p_pos = p.pos(0);
        auto p_rot = p.R(0);

        auto normal = p.normal();

        // plan2sphs.n
        auto plan_to_sphere = xt::eval(xt::linalg::dot(s_pos - p_pos, normal));
        auto sign = xt::sign(plan_to_sphere);

        xt::view(pts, 0) = s_pos - sign*s.radius()*normal;
        xt::view(pts, 1) = s_pos - plan_to_sphere*normal;
        return pts;
    }

    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const plan<dim, false>)
    {
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }
}