#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <fmt/color.h>
#include <fmt/format.h>

#include "../contact/property.hpp"
#include "../utils.hpp"

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
    template <std::size_t dim, class problem_t_>
    struct neighbor
    {
        using problem_t = problem_t_;
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
        /**
         * @brief The s for contact \c i \c j in fixed point algorithm
         */
        double sij;

        contact_property<problem_t> property;

        auto to_json() const
        {
            return nl::json{
                {"i",        i                 },
                {"j",        j                 },
                {"pi",       pi                },
                {"pj",       pj                },
                {"normal",   nij               },
                {"distance", dij               },
                {"property", property.to_json()}
            };
        }
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
    template <std::size_t dim, class problem_t>
    std::ostream& operator<<(std::ostream& out, const neighbor<dim, problem_t>& neigh)
    {
        out << fmt::format(fg(fmt::color::steel_blue) | fmt::emphasis::bold, "contact") << std::endl;
        print_indented(out, 4, "{:<12} : {}", "i", neigh.i);
        print_indented(out, 4, "{:<12} : {}", "j", neigh.j);
        print_indented(out, 4, "{:<12} : {}", "pi", neigh.pi);
        print_indented(out, 4, "{:<12} : {}", "pj", neigh.pj);
        print_indented(out, 4, "{:<12} : {}", "normal", neigh.nij);
        print_indented(out, 4, "{:<12} : {}", "distance", neigh.dij);
        print_indented(out, 4, "{:<12} :", "property");
        to_stream(out, 8, neigh.property);
        return out;
    }
}
