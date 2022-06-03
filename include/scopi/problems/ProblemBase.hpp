#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../objects/neighbor.hpp"

namespace scopi
{
    class ProblemBase
    {
    protected:
        ProblemBase(std::size_t nparts, double dt);

        template<std::size_t dim>
        void matrix_free_gemv_inv_P(const scopi_container<dim>& particles,
                                    xt::xtensor<double, 1>& U,
                                    std::size_t active_offset,
                                    std::size_t row);

        std::size_t m_nparticles;
        double m_dt;

        std::vector<int> m_A_rows;
        std::vector<int> m_A_cols;
        std::vector<double> m_A_values;
        xt::xtensor<double, 1> m_distances;
    private:
        void matrix_free_gemv_inv_P_moment(const scopi_container<2>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);
        void matrix_free_gemv_inv_P_moment(const scopi_container<3>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);
    };

    template<std::size_t dim>
    void ProblemBase::matrix_free_gemv_inv_P(const scopi_container<dim>& particles,
                                             xt::xtensor<double, 1>& U,
                                             std::size_t active_offset,
                                             std::size_t row)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            U(3*row + d) /= (-1.*particles.m()(active_offset + row)); 
        }
        matrix_free_gemv_inv_P_moment(particles, U, active_offset, row);
    }


}
