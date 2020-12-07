#pragma once

#include "sphere.hpp"
#include "globule.hpp"
#include "plan.hpp"

namespace scopi
{
    template<std::size_t dim>
    double distance(const sphere<dim, false>, const sphere<dim, false>)
    {
        return 1;
    }

    template<std::size_t dim>
    double distance(const plan<dim, false>, const plan<dim, false>)
    {
        return 2;
    }

    template<std::size_t dim>
    double distance(const globule<dim, false>, const globule<dim, false>)
    {
        return 3;
    }


    template<std::size_t dim>
    double distance(const sphere<dim, false>, const globule<dim, false>)
    {
        return 4;
    }

    template<std::size_t dim>
    double distance(const sphere<dim, false>, const plan<dim, false>)
    {
        return 5;
    }

    template<std::size_t dim>
    double distance(const globule<dim, false>, const plan<dim, false>)
    {
        return 6;
    }
}