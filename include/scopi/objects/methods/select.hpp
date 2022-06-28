#pragma once

#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
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

#include "nlohmann/json.hpp"

namespace nl = nlohmann;

namespace scopi
{

    struct index
    {
        index(std::size_t ii)
        : i(ii)
        {}
        std::size_t i;
    };

    // SPHERE
    template<std::size_t dim>
    std::unique_ptr<object<dim, false>> select_object(const sphere<dim, false>& s, const std::size_t)
    {
        return std::make_unique<sphere<dim, false>>(s);
    }


    // SUPERELLIPSOID
    template<std::size_t dim>
    std::unique_ptr<object<dim, false>> select_object(const superellipsoid<dim, false>& s, const std::size_t)
    {
        return std::make_unique<superellipsoid<dim, false>>(s);
    }

    // PLAN
    template<std::size_t dim>
    std::unique_ptr<object<dim, false>> select_object(const plan<dim, false>& s, const std::size_t)
    {
        return std::make_unique<plan<dim, false>>(s);
    }

    // WORM
    template<std::size_t dim>
    std::unique_ptr<object<dim, false>> select_object(const worm<dim, false>& s, const std::size_t i)
    {
        return s.get_sphere(i);
    }

    template <std::size_t dim>
    struct select_object_functor
    {
        using return_type = std::unique_ptr<object<dim, false>>;

        template <class T>
        return_type run(const T& obj, const index& idx) const
        {
            return select_object(obj, idx.i);
        }

        return_type on_error(const object<dim, false>&, const index&) const
        {
            return {};
        }
    };

    template <std::size_t dim>
    using select_object_dispatcher = double_static_dispatcher
    <
        select_object_functor<dim>,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const worm<dim, false>,
                    const plan<dim, false>>,
        typename select_object_functor<dim>::return_type,
        antisymmetric_dispatch, const index, mpl::vector<const index>
    >;
}
