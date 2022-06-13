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
#include "../types/globule.hpp"
#include "../types/plan.hpp"
#include "../neighbor.hpp"
#include "../dispatch.hpp"

namespace scopi
{

    // SPHERE
    template<std::size_t dim>
    std::size_t number_contact_per_particle(const sphere<dim, false>&)
    {
        return 0;
    }


    // SUPERELLIPSOID
    template<std::size_t dim>
    std::size_t number_contact_per_particle(const superellipsoid<dim, false>&)
    {
        return 0;
    }

    // PLAN
    template<std::size_t dim>
    std::size_t number_contact_per_particle(const plan<dim, false>)
    {
        return 0;
    }

    // GLOBULE
    template<std::size_t dim>
    std::size_t number_contact_per_particle(const globule<dim, false>)
    {
        return 5;
    }

    template <std::size_t dim>
    struct number_contact_functor
    {
        using return_type = std::size_t;

        template <class T1>
        return_type run(const T1& obj1) const
        {
            return number_contact_per_particle(obj1);
        }

        return_type on_error(const object<dim, false>&) const
        {
            return 0;
        }
    };

    template <std::size_t dim>
    using number_contact_per_particle_dispatcher = unit_static_dispatcher
    <
        number_contact_functor<dim>,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const globule<dim, false>,
                    const plan<dim, false>>,
        typename number_contact_functor<dim>::return_type
    >;
}
