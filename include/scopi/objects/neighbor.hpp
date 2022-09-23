#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

namespace scopi
{
    /**
     * @brief Structure of a neighbor.
     *
     * Particles \c i and \c j are in contact.
     * \c pi = \c pj if the two particles are in contact.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    struct neighbor
    {
        /**
         * @brief Index of the particle \c i.
         */
        std::size_t i;
        /**
         * @brief Index of the particle \c j.
         */
        std::size_t j;
        /**
         * @brief Distance between particles \c i and \c j;
         */
        double dij;
        /**
         * @brief Outer normal vector to particle \c j.
         */
        xt::xtensor_fixed<double, xt::xshape<dim>> nij;
        /**
         * @brief Point in particle \c i which realizes the distance between the two particles.
         */
        xt::xtensor_fixed<double, xt::xshape<dim>> pi;
        /**
         * @brief Point in particle \c j which realizes the distance between the two particles.
         */
        xt::xtensor_fixed<double, xt::xshape<dim>> pj;
    };

    /**
     * @brief 
     *
     * \todo Write documentation.
     *
     * @tparam dim
     * @param out
     * @param neigh
     *
     * @return 
     */
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
