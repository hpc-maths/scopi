#pragma once

#include <xtensor-blas/xlinalg.hpp>

#include "../dispatch.hpp"
#include "../types/globule.hpp"
#include "../types/plane.hpp"
#include "../types/sphere.hpp"
#include "../types/superellipsoid.hpp"

namespace scopi
{
    // SUPERELLIPSOID - SUPERELLIPSOID
    template <std::size_t dim>
    double distance(const superellipsoid<dim, false> s1, const superellipsoid<dim, false> s2)
    {
        std::cout << "distance : SUPERELLIPSOID - SUPERELLIPSOID" << std::endl;
        return 10;
    }

    // SPHERE - SPHERE
    template <std::size_t dim>
    double distance(const sphere<dim, false> s1, const sphere<dim, false> s2)
    {
        std::cout << "distance : SPHERE - SPHERE" << std::endl;
        return xt::linalg::norm(s1.pos() - s2.pos()) - s1.radius() - s2.radius();
    }

    // PLANE - PLANE
    template <std::size_t dim>
    double distance(const plane<dim, false>, const plane<dim, false>)
    {
        std::cout << "distance : PLANE - PLANE" << std::endl;
        return 20;
    }

    // GLOBULE - GLOBULE
    template <std::size_t dim>
    double distance(const globule<dim, false>, const globule<dim, false>)
    {
        std::cout << "distance : GLOBULE - GLOBULE" << std::endl;
        return 30;
    }

    // SPHERE - SUPERELLIPSOID
    template <std::size_t dim>
    double distance(const sphere<dim, false>, const superellipsoid<dim, false>)
    {
        std::cout << "distance : SPHERE - SUPERELLIPSOID" << std::endl;
        return 40;
    }

    // SPHERE - GLOBULE
    template <std::size_t dim>
    double distance(const sphere<dim, false>, const globule<dim, false>)
    {
        std::cout << "distance : SPHERE - GLOBULE" << std::endl;
        return 50;
    }

    // SPHERE - PLANE
    template <std::size_t dim>
    double distance(const sphere<dim, false>, const plane<dim, false>)
    {
        std::cout << "distance : SPHERE - PLANE" << std::endl;
        return 60;
    }

    // SUPERELLIPSOID - GLOBULE
    template <std::size_t dim>
    double distance(const superellipsoid<dim, false>, const globule<dim, false>)
    {
        std::cout << "distance : SUPERELLIPSOID - GLOBULE" << std::endl;
        return 70;
    }

    // SUPERELLIPSOID - PLANE
    template <std::size_t dim>
    double distance(const superellipsoid<dim, false>, const plane<dim, false>)
    {
        std::cout << "distance : SUPERELLIPSOID - PLANE" << std::endl;
        return 80;
    }

    // GLOBULE - PLANE
    template <std::size_t dim>
    double distance(const globule<dim, false>, const plane<dim, false>)
    {
        std::cout << "distance : GLOBULE - PLANE" << std::endl;
        return 90;
    }

    struct distance_functor
    {
        using return_type = double;

        template <class T1, class T2>
        return_type run(const T1& obj1, const T2& obj2) const
        {
            return distance(obj1, obj2);
        }

        template <std::size_t dim>
        return_type on_error(const object<dim, false>&, const object<dim, false>&) const
        {
            return 0;
        }
    };

    template <std::size_t dim>
    using distance_dispatcher = double_static_dispatcher<
        distance_functor,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>, const superellipsoid<dim, false>, const globule<dim, false>, const plane<dim, false>>,
        typename distance_functor::return_type,
        symmetric_dispatch>;
}
