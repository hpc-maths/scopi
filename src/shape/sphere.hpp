#pragma once

#include <vector>

#include "base.hpp"

namespace scopi
{
    class position{};
    class quaternion{};

    class sphere: shape_base<sphere>
    {
      public:
        sphere(const position& pos, double radius, const quaternion& q)
          : m_pos{pos}, m_radius{radius}, m_q{q}
        {}

      private:
        position m_pos;
        double m_radius;
        quaternion m_q;
    };

    template<class... T>
    class globule: object_base<globule<T...>, T...>
    {
        using base_type = object_base<globule<T...>, T...>;
        using base_type::m_shape;
      public:
        globule(const T&... shape): base_type{shape...}
        {}

    };

    template<class D, class... T>
    auto make_object(T... s)
    {
        return object_base<D<T...>, T...>(s...);
    }
}