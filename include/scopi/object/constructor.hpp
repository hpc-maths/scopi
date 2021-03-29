#pragma once

#include <array>
#include <memory>
#include <tuple>

#include <xtensor/xfixed.hpp>

namespace scopi
{
    ///////////////////////////////////
    // object_constructor definition //
    ///////////////////////////////////
    template<std::size_t dim, bool owner>
    class object;

    template<std::size_t dim>
    struct base_constructor
    {
        virtual ~base_constructor() = default;
        virtual std::unique_ptr<object<dim, false>> operator()(xt::xtensor_fixed<double, xt::xshape<dim>>* pos, xt::xtensor_fixed<double, xt::xshape<dim, dim>>* r) const = 0;
    };

    template<class T, class... Args>
    class object_constructor: public base_constructor<T::dim>
    {
    public:
        static constexpr std::size_t dim = T::dim;
        using object_type = T;
        using tuple_type = std::tuple<Args...>;

        template<class... CTA>
        object_constructor(CTA&&... args);

        virtual std::unique_ptr<object<dim, false>> operator()(xt::xtensor_fixed<double, xt::xshape<dim>>* pos, xt::xtensor_fixed<double, xt::xshape<dim, dim>>* r) const override;

    private:

        template<std::size_t... I>
        auto constructor(xt::xtensor_fixed<double, xt::xshape<dim>>* pos, xt::xtensor_fixed<double, xt::xshape<dim, dim>>* r, std::index_sequence<I...>) const;

        tuple_type m_extra;
    };

     ///////////////////////////////////////
    // object_constructor implementation //
    ///////////////////////////////////////
    template<class T, class... Args>
    template<class... CTA>
    object_constructor<T, Args...>::object_constructor(CTA&&... args)
    : m_extra(std::forward<CTA>(args)...)
    {}

    template<class T, class... Args>
    auto object_constructor<T, Args...>::operator()(xt::xtensor_fixed<double, xt::xshape<dim>>* pos, xt::xtensor_fixed<double, xt::xshape<dim, dim>>* r) const -> std::unique_ptr<object<dim, false>>
    {
        return constructor(pos, r, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<class T, class... Args>
    template<std::size_t... I>
    auto object_constructor<T, Args...>::constructor(xt::xtensor_fixed<double, xt::xshape<dim>>* pos, xt::xtensor_fixed<double, xt::xshape<dim, dim>>* r, std::index_sequence<I...>) const
    {
        return std::make_unique<object_type>(pos, r, std::get<I>(m_extra)...);
    }

    template<class T, class... Args>
    auto make_object_constructor(Args&&... args)
    {
        using constructor_type = object_constructor<T, Args...>;
        return std::make_unique<constructor_type>(args...);
    }
}
