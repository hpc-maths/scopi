#pragma once

#include <xtl/xmultimethods.hpp>

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

        template <class lhs_type, class... Args>
        static return_type invoke_executor(lhs_type& lhs, Args&&... args)
        {
            executor exec;
            return exec.run(lhs, std::forward<Args>(args)...);
        }

        template <class... Args>
        static return_type dispatch_lhs(base_lhs& lhs, mpl::vector<>, Args&&... args)
        {
            executor exec;
            return exec.on_error(lhs, std::forward<Args>(args)...);
        }

        template <class T, class... U, class... Args>
        static return_type dispatch_lhs(base_lhs& lhs, mpl::vector<T, U...>, Args&&... args)
        {
            if (T* p = dynamic_cast<T*>(&lhs))
            {
                return invoke_executor(*p, std::forward<Args>(args)...);
            }
            return dispatch_lhs(lhs, mpl::vector<U...>(), std::forward<Args>(args)...);
        }

    public:

        template <class... Args>
        static return_type dispatch(base_lhs& lhs, Args&&... args)
        {
            return dispatch_lhs(lhs, lhs_type_list(), std::forward<Args>(args)...);
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

        template <class lhs_type, class rhs_type, class... Args>
        static return_type invoke_executor(lhs_type& lhs,
                                           rhs_type& rhs,
                                           std::false_type,
                                           Args&&... args)
        {
            executor exec;
            return exec.run(lhs, rhs, std::forward<Args>(args)...);
        }

        template <class lhs_type, class rhs_type, class... Args>
        static return_type invoke_executor(lhs_type& lhs,
                                           rhs_type& rhs,
                                           std::true_type,
                                           Args&&... args)
        {
            executor exec;
            return exec.run(rhs, lhs, std::forward<Args>(args)...);
        }

        template <class lhs_type, class... Args>
        static return_type dispatch_rhs(lhs_type& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<>,
                                        Args&&... args)
        {
            executor exec;
            return exec.on_error(lhs, rhs, std::forward<Args>(args)...);
        }

        template <class lhs_type, class T, class... U, class... Args>
        static return_type dispatch_rhs(lhs_type& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<T, U...>,
                                        Args&&... args)
        {
            if (T* p = dynamic_cast<T*>(&rhs))
            {
                constexpr size_t lhs_index = mpl::index_of<lhs_type_list, lhs_type>::value;
                constexpr size_t rhs_index = mpl::index_of<rhs_type_list, T>::value;

                using invoke_flag = std::integral_constant<bool,
                    std::is_same<symmetric, symmetric_dispatch>::value && (rhs_index < lhs_index)>;
                return invoke_executor(lhs, *p, invoke_flag(), std::forward<Args>(args)...);
            }
            return dispatch_rhs(lhs, rhs, mpl::vector<U...>(), std::forward<Args>(args)...);
        }

        template <class... Args>
        static return_type dispatch_lhs(base_lhs& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<>,
                                        Args&&... args)
        {
            executor exec;
            return exec.on_error(lhs, rhs, std::forward<Args>(args)...);
        }

        template <class T, class... U, class... Args>
        static return_type dispatch_lhs(base_lhs& lhs,
                                        base_rhs& rhs,
                                        mpl::vector<T, U...>,
                                        Args&&... args)
        {
            if (T* p = dynamic_cast<T*>(&lhs))
            {
                return dispatch_rhs(*p, rhs, rhs_type_list(), std::forward<Args>(args)...);
            }
            return dispatch_lhs(lhs, rhs, mpl::vector<U...>(), std::forward<Args>(args)...);
        }

    public:

        template <class... Args>
        static return_type dispatch(base_lhs& lhs, base_rhs& rhs, Args&&... args)
        {
            return dispatch_lhs(lhs, rhs, lhs_type_list(), std::forward<Args>(args)...);
        }
    };

}
