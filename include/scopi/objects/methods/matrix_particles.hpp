#pragma once

#include <cstddef>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include "../types/sphere.hpp"
#include "../types/superellipsoid.hpp"
#include "../types/worm.hpp"
#include "../types/plan.hpp"
#include "../neighbor.hpp"
#include "../dispatch.hpp"

namespace scopi
{
    // SPHERE
    template<std::size_t dim>
    xt::xtensor<double, 1> distances_per_particle(const sphere<dim, false>&)
    {
        return xt::xtensor<double, 1>({});
    }


    // SUPERELLIPSOID
    template<std::size_t dim>
    xt::xtensor<double, 1> distances_per_particle(const superellipsoid<dim, false>&)
    {
        return xt::xtensor<double, 1>({});
    }

    // PLAN
    template<std::size_t dim>
    xt::xtensor<double, 1> distances_per_particle(const plan<dim, false>&)
    {
        return xt::xtensor<double, 1>({});
    }

    // WORM
    template<std::size_t dim>
    xt::xtensor<double, 1> distances_per_particle(const worm<dim, false>& g)
    {
        xt::xtensor<double, 1>::shape_type shape = {g.size()-1};
        xt::xtensor<double, 1> distances(shape);
        // TODO use xtensor's functions instead of a loop
        for (std::size_t i = 0; i < g.size()-1; ++i)
        {
            auto si_pos = g.pos(i);
            auto sj_pos = g.pos(i+1);
            auto si_to_sj = (sj_pos - si_pos)/xt::linalg::norm(sj_pos - si_pos);
            auto pi = si_pos + g.radius()*si_to_sj;
            auto pj = sj_pos - g.radius()*si_to_sj;
            auto nij = (pj - sj_pos)/xt::linalg::norm(pj - sj_pos);
            auto dij = xt::linalg::dot(pi - pj, nij)[0];
            distances(i) = -dij;
        }
        return distances;
    }

    template <std::size_t dim>
    struct distances_per_particle_functor
    {
        using return_type = xt::xtensor<double, 1>;

        template <class T1>
        return_type run(const T1& obj1) const
        {
            return distances_per_particle(obj1);
        }

        return_type on_error(const object<dim, false>&) const
        {
            return xt::xtensor<double, 1>({});
        }
    };

    template <std::size_t dim>
    using distances_per_particle_dispatcher = unit_static_dispatcher
    <
        distances_per_particle_functor<dim>,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const worm<dim, false>,
                    const plan<dim, false>>,
        typename distances_per_particle_functor<dim>::return_type
    >;



    // SPHERE
    template<std::size_t dim>
    xt::xtensor<double, 2> matrix_per_particle(const sphere<dim, false>&)
    {
        return xt::xtensor<double, 2>({});
    }


    // SUPERELLIPSOID
    template<std::size_t dim>
    xt::xtensor<double, 2> matrix_per_particle(const superellipsoid<dim, false>&)
    {
        return xt::xtensor<double, 2>({});
    }

    // PLAN
    template<std::size_t dim>
    xt::xtensor<double, 2> matrix_per_particle(const plan<dim, false>&)
    {
        return xt::xtensor<double, 2>({});
    }

    // WORM
    template<std::size_t dim>
    xt::xtensor<double, 2> matrix_per_particle(const worm<dim, false>& g)
    {
        xt::xtensor<double, 2>::shape_type shape = {12*(g.size()-1), 3};
        xt::xtensor<double, 2> mat(shape);
        std::size_t index = 0;
        for (std::size_t i = 0; i < g.size()-1; ++i)
        {
            auto si_pos = g.pos(i);
            auto sj_pos = g.pos(i+1);
            auto si_to_sj = (sj_pos - si_pos)/xt::linalg::norm(sj_pos - si_pos);
            auto pi = si_pos + g.radius()*si_to_sj;
            auto pj = sj_pos - g.radius()*si_to_sj;
            auto nij = (pj - sj_pos)/xt::linalg::norm(pj - sj_pos);

            // D <= 0, cols u
            for (std::size_t d = 0; d < 3; ++d)
            {
                mat(index, 0) = 2*i+1;
                mat(index, 1) = i*3 + d;
                mat(index, 2) = nij[d];
                index++;
            }
            for (std::size_t d = 0; d < 3; ++d)
            {
                mat(index, 0) = 2*i+1;
                mat(index, 1) = (i+1)*3 + d;
                mat(index, 2) = - nij[d];
                index++;
            }

            auto ri_cross = cross_product<dim>(pi - g.pos(i));
            auto rj_cross = cross_product<dim>(pj - g.pos(i+1));
            auto Ri = rotation_matrix<3>(g.q(i));
            auto Rj = rotation_matrix<3>(g.q(i+1));

            // D <= 0, cols w
            auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                mat(index, 0) = 2*i+1;
                mat(index, 1) = 3*i + ip;
                mat(index, 2) = -(nij[0]*dot(0, ip)+nij[1]*dot(1, ip)+nij[2]*dot(2, ip));
                index++;
            }

            dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                mat(index, 0) = 2*i;
                mat(index, 1) = 3*(i+1) + ip;
                mat(index, 2) = - (nij[0]*dot(0, ip)+nij[1]*dot(1, ip)+nij[2]*dot(2, ip));
                index++;
            }
        }
        return mat;
    }

    template <std::size_t dim>
    struct matrix_per_particle_functor
    {
        using return_type = xt::xtensor<double, 2>;

        template <class T1>
        return_type run(const T1& obj1) const
        {
            return matrix_per_particle(obj1);
        }

        return_type on_error(const object<dim, false>&) const
        {
            return xt::xtensor<double, 2>({});
        }
    };

    template <std::size_t dim>
    using matrix_per_particle_dispatcher = unit_static_dispatcher
    <
        matrix_per_particle_functor<dim>,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const worm<dim, false>,
                    const plan<dim, false>>,
        typename matrix_per_particle_functor<dim>::return_type
    >;
}
