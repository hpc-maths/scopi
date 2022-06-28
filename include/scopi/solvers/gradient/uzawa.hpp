#pragma once

#ifdef SCOPI_USE_MKL
#include <mkl_spblas.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include "projection_max.hpp"

namespace scopi{
    template<class projection_t = projection_max>
    class uzawa: public projection_t
    {
    protected:
        uzawa(std::size_t max_iter, double rho, double tol_dg, double tol_l);
        std::size_t projection(const sparse_matrix_t& A, const struct matrix_descr& descr, const xt::xtensor<double, 1>& c, xt::xtensor<double, 1>& l);
    private:
        std::size_t m_max_iter;
        double m_rho;
        double m_tol_dg;
        double m_tol_l;

        sparse_status_t m_status;
        xt::xtensor<double, 1> m_dg;
        xt::xtensor<double, 1> m_uu;
    };

    template<class projection_t>
    uzawa<projection_t>::uzawa(std::size_t max_iter, double rho, double tol_dg, double tol_l)
    : projection_t()
    , m_max_iter(max_iter)
    , m_rho(rho)
    , m_tol_dg(tol_dg)
    , m_tol_l(tol_l)
    {}

    template<class projection_t>
    std::size_t uzawa<projection_t>::projection(const sparse_matrix_t& A, const struct matrix_descr& descr, const xt::xtensor<double, 1>& c, xt::xtensor<double, 1>& l)
    {
        std::size_t iter = 0;
        while (iter < m_max_iter)
        {
            xt::noalias(m_dg) = c;

            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, l.data(), 1., m_dg.data());
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for dg = A*l+dg: " << m_status;

            xt::noalias(l) = this->projection_cone(l - m_rho * m_dg);
            double norm_dg = xt::amax(xt::abs(m_dg))(0);
            double norm_l = xt::amax(xt::abs(l))(0);

            // uu = A*l + c
            // TODO for performance reasons, do that only if verbose mode
            xt::noalias(m_uu) = c;
            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, l.data(), 1., m_uu.data());
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for uu = A*l+c: " << m_status;
            double constraint = double((xt::amin(m_uu))(0));
            PLOG_VERBOSE << constraint;

            if (norm_dg < m_tol_dg || norm_l < m_tol_l)
            {
                return iter+1;
            }
            iter++;
        }
        PLOG_ERROR << "Uzawa does not converge";
        return iter;
    }
}
#endif
