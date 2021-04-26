#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>

#include "sphere.hpp"
#include "globule.hpp"
#include "plan.hpp"
#include "neighbor.hpp"

namespace scopi
{
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& si, const sphere<dim, false>& sj)
    {
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto si_pos = xt::view(si.pos(), 0);
        auto sj_pos = xt::view(sj.pos(), 0);
        auto si_to_sj = (sj_pos - si_pos)/xt::linalg::norm(sj_pos - si_pos);

        neighbor<dim> neigh;
        neigh.pi = si_pos + si.radius()*si_to_sj;
        neigh.pj = sj_pos - sj.radius()*si_to_sj;
        neigh.nj = (neigh.pj - sj_pos)/xt::linalg::norm(neigh.pj - sj_pos);
        neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nj)[0];
        return neigh;
    }

    template<std::size_t dim>
    auto closest_points(const plan<dim, false>, const plan<dim, false>)
    {
        return neighbor<dim>();
    }

    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const globule<dim, false>)
    {
        return neighbor<dim>();
    }

    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>, const globule<dim, false>)
    {
        return neighbor<dim>();
    }

    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s, const plan<dim, false>& p)
    {
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto s_pos = s.pos(0);
        auto p_pos = p.pos(0);

        auto normal = p.normal();

        // plan2sphs.n
        auto plan_to_sphere = xt::eval(xt::linalg::dot(s_pos - p_pos, normal));
        auto sign = xt::sign(plan_to_sphere);

        neighbor<dim> neigh;
        neigh.pi = s_pos - sign*s.radius()*normal;
        neigh.pj = s_pos - plan_to_sphere*normal;
        neigh.nj = sign*normal;
        neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nj)[0];
        return neigh;
    }

    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const plan<dim, false>)
    {
        return neighbor<dim>();
    }
}