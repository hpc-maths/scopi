#pragma once

#include <xtensor-blas/xlinalg.hpp>

#include "sphere.hpp"
#include "superellipsoid.hpp"
#include "globule.hpp"
#include "plan.hpp"

namespace scopi
{
    // SUPERELLIPSOID - SUPERELLIPSOID
    template<std::size_t dim>
    double distance(const superellipsoid<dim, false> s1, const superellipsoid<dim, false> s2)
    {
        std::cout << "distance : SUPERELLIPSOID - SUPERELLIPSOID" << std::endl;
        std::cout << "rotation s1 : " << s1.rotation() << std::endl;
        std::cout << "rotation s2 : " << s2.rotation() << std::endl;
        return 10;
    }

    // SPHERE - SPHERE
    template<std::size_t dim>
    double distance(const sphere<dim, false> s1, const sphere<dim, false> s2)
    {
        std::cout << "distance : SPHERE - SPHERE" << std::endl;
        return xt::linalg::norm(s1.pos() - s2.pos()) - s1.radius() - s2.radius();
    }

    // PLAN - PLAN
    template<std::size_t dim>
    double distance(const plan<dim, false>, const plan<dim, false>)
    {
        std::cout << "distance : PLAN - PLAN" << std::endl;
        return 20;
    }

    // GLOBULE - GLOBULE
    template<std::size_t dim>
    double distance(const globule<dim, false>, const globule<dim, false>)
    {
        std::cout << "distance : GLOBULE - GLOBULE" << std::endl;
        return 30;
    }

    // SPHERE - SUPERELLIPSOID
    template<std::size_t dim>
    double distance(const sphere<dim, false>, const superellipsoid<dim, false>)
    {
        std::cout << "distance : SPHERE - SUPERELLIPSOID" << std::endl;
        return 40;
    }

    // SPHERE - GLOBULE
    template<std::size_t dim>
    double distance(const sphere<dim, false>, const globule<dim, false>)
    {
        std::cout << "distance : SPHERE - GLOBULE" << std::endl;
        return 50;
    }

    // SPHERE - PLAN
    template<std::size_t dim>
    double distance(const sphere<dim, false>, const plan<dim, false>)
    {
        std::cout << "distance : SPHERE - PLAN" << std::endl;
        return 60;
    }

    // SUPERELLIPSOID - GLOBULE
    template<std::size_t dim>
    double distance(const superellipsoid<dim, false>, const globule<dim, false>)
    {
        std::cout << "distance : SUPERELLIPSOID - GLOBULE" << std::endl;
        return 70;
    }

    // SUPERELLIPSOID - PLAN
    template<std::size_t dim>
    double distance(const superellipsoid<dim, false>, const plan<dim, false>)
    {
        std::cout << "distance : SUPERELLIPSOID - PLAN" << std::endl;
        return 80;
    }

    // GLOBULE - PLAN
    template<std::size_t dim>
    double distance(const globule<dim, false>, const plan<dim, false>)
    {
        std::cout << "distance : GLOBULE - PLAN" << std::endl;
        return 90;
    }

}
