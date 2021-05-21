#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace scopi
{
    template<std::size_t dim>
    struct neighbor
    {
        std::size_t i, j;
        double dij;
        xt::xtensor_fixed<double, xt::xshape<dim>> nij;
        xt::xtensor_fixed<double, xt::xshape<dim>> pi, pj;
    };

    template<std::size_t dim>
    std::ostream& operator<<(std::ostream& out, const neighbor<dim>& neigh)
    {
        out << "neighbor:" << std::endl;
        out << "\ti: " << neigh.i << std::endl;
        out << "\tj: " << neigh.j << std::endl;
        out << "\tpi: " << neigh.pi << std::endl;
        out << "\tpj: " << neigh.pj << std::endl;
        out << "\tnij: " << neigh.nij << std::endl;
        out << "\tdij: " << neigh.dij;
        return out;
    }
}