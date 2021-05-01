#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace scopi
{
    template<std::size_t dim>
    struct neighbor
    {
        double dij;
        xt::xtensor_fixed<double, xt::xshape<dim>> nj;
        xt::xtensor_fixed<double, xt::xshape<dim>> pi, pj;
    };

    template<std::size_t dim>
    std::ostream& operator<<(std::ostream& out, const neighbor<dim>& neigh)
    {
        out << "neighbor:" << std::endl;
        out << "\tpi: " << neigh.pi << std::endl;
        out << "\tpj: " << neigh.pj << std::endl;
        out << "\tnj: " << neigh.nj << std::endl;
        out << "\tdij: " << neigh.dij;
        return out;
    }
}