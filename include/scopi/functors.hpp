#pragma once

#include "dispatch.hpp"

#include "object/distance.hpp"

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
                    const globule<dim, false>,
                    const plan<dim, false>>,
        typename distance_functor::return_type,
        symmetric_dispatch
    >;
}