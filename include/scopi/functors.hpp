#pragma once

#include <xtensor/xfixed.hpp>

#include "dispatch.hpp"

#include "object/distance.hpp"
#include "object/closest_points.hpp"
#include "object/neighbor.hpp"

namespace scopi
{
    struct print_functor
    {
        using return_type = void;

        template <class T>
        static return_type run(const T& obj)
        {
            obj.print();
        }

        template<std::size_t dim>
        static return_type on_error(const object<dim, false>&)
        {
            std::cout << "unknown object print function" << std::endl;
        }
    };

    template <std::size_t dim>
    using print_dispatcher = unit_static_dispatcher
    <
        print_functor,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const globule<dim, false>,
                    const plan<dim, false>>,
        typename print_functor::return_type
    >;

    struct distance_functor
    {
        using return_type = double;

        template <class T1, class T2>
        return_type run(const T1& obj1, const T2& obj2) const
        {
            return distance(obj1, obj2);
        }

        template<std::size_t dim>
        return_type on_error(const object<dim, false>&, const object<dim, false>&) const
        {
            return 0;
        }
    };

    template <std::size_t dim>
    using distance_dispatcher = double_static_dispatcher
    <
        distance_functor,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const globule<dim, false>,
                    const plan<dim, false>>,
        typename distance_functor::return_type,
        symmetric_dispatch
    >;

    template <std::size_t dim>
    struct closest_points_functor
    {
        using return_type = neighbor<dim>;

        template <class T1, class T2>
        return_type run(const T1& obj1, const T2& obj2) const
        {
            return closest_points(obj1, obj2);
        }

        return_type on_error(const object<dim, false>&, const object<dim, false>&) const
        {
            return {};
        }
    };

    template <std::size_t dim>
    using closest_points_dispatcher = double_static_dispatcher
    <
        closest_points_functor<dim>,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>,
                    const superellipsoid<dim, false>,
                    const globule<dim, false>,
                    const plan<dim, false>>,
        typename closest_points_functor<dim>::return_type,
        symmetric_dispatch
    >;
}
