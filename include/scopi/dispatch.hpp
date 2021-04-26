#pragma once

#include <xtl/xmultimethods.hpp>

#include "object/sphere.hpp"
#include "object/superellipsoid.hpp"
#include "object/globule.hpp"
#include "object/plan.hpp"

namespace scopi
{
    namespace mpl = xtl::mpl;

    struct symmetric_dispatch {};
    struct antisymmetric_dispatch {};

    /*********************
     * static_dispatcher *
     *********************/

    template
    <
        class executor,
        class base_lhs,
        class lhs_type_list,
        class return_type = void
    >
    class unit_static_dispatcher
    {
    private:

        template <class lhs_type>
        static return_type invoke_executor(lhs_type& lhs)
        {
            executor exec;
            return exec.run(lhs);
        }

        static return_type dispatch_lhs(base_lhs& lhs,
                                        mpl::vector<>)
        {
            executor exec;
            return exec.on_error(lhs);
        }

        template <class T, class... U>
        static return_type dispatch_lhs(base_lhs& lhs,
                                        mpl::vector<T, U...>)
        {
            if (T* p = dynamic_cast<T*>(&lhs))
            {
                return invoke_executor(*p);
            }
            return dispatch_lhs(lhs, mpl::vector<U...>());
        }

    public:

        static return_type dispatch(base_lhs& lhs)
        {
            return dispatch_lhs(lhs, lhs_type_list());
        }
    };

    template
    <
        class executor,
        class base_lhs,
        class lhs_type_list,
        class return_type = void,
        class symmetric = antisymmetric_dispatch,
        class base_rhs = base_lhs,
        class rhs_type_list = lhs_type_list
    >
    class double_static_dispatcher
    {
    private:

        template <class lhs_type, class rhs_type>
        static return_type invoke_executor(lhs_type& lhs,
                                           rhs_type& rhs,
                                           std::false_type)
        {
            executor exec;
            return exec.run(lhs, rhs);
        }

        template <class lhs_type, class rhs_type>
        static return_type invoke_executor(lhs_type& lhs,
                                           rhs_type& rhs,
                                           std::true_type)
        {
            executor exec;
            return exec.run(rhs, lhs);
        }

        template <class lhs_type>
        static return_type dispatch_rhs(lhs_type& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<>)
        {
            executor exec;
            return exec.on_error(lhs, rhs);
        }

        template <class lhs_type, class T, class... U>
        static return_type dispatch_rhs(lhs_type& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<T, U...>)
        {
            if (T* p = dynamic_cast<T*>(&rhs))
            {
                constexpr size_t lhs_index = mpl::index_of<lhs_type_list, lhs_type>::value;
                constexpr size_t rhs_index = mpl::index_of<rhs_type_list, T>::value;

                using invoke_flag = std::integral_constant<bool,
                    std::is_same<symmetric, symmetric_dispatch>::value && (rhs_index < lhs_index)>;
                return invoke_executor(lhs, *p, invoke_flag());
            }
            return dispatch_rhs(lhs, rhs, mpl::vector<U...>());
        }

        static return_type dispatch_lhs(base_lhs& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<>)
        {
            executor exec;
            return exec.on_error(lhs, rhs);
        }

        template <class T, class... U>
        static return_type dispatch_lhs(base_lhs& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<T, U...>)
        {
            if (T* p = dynamic_cast<T*>(&lhs))
            {
                return dispatch_rhs(*p, rhs, rhs_type_list());
            }
            return dispatch_lhs(lhs, rhs, mpl::vector<U...>());
        }

    public:

        static return_type dispatch(base_lhs& lhs, base_rhs& rhs)
        {
            return dispatch_lhs(lhs, rhs, lhs_type_list());
        }
    };

}
