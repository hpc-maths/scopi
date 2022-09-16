#pragma once

#ifdef SCOPI_USE_MKL
#include <mkl_spblas.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../projection.hpp"
#include "../../problems/DryWithoutFriction.hpp"

namespace scopi{
    /**
     * @brief Accelerated Projected Gradient Descent algorithm with Adaptive Restart.
     *
     * See OptimProjectedGradient for the notations.
     * The algorithm is
     *  - \f$ \indexUzawa = 0 \f$;
     *  - \f$ \mathbf{l}^{\indexUzawa} = 0 \f$;
     *  - \f$ \mathbf{y}^{\indexUzawa} = 0 \f$;
     *  - \f$ \theta^{\indexUzawa} = 1 \f$.
     *  - While (\f$ \convergenceCriterion \f$)
     *      - \f$ \mathbf{dg}^{\indexUzawa} = \mathbb{A} \mathbf{y}^{\indexUzawa} + \mathbf{e} \f$;
     *      - \f$ \mathbf{l}^{\indexUzawa+1} = \max \left (\mathbf{y}^{\indexUzawa} - \rho \mathbf{dg}^{\indexUzawa}, 0 \right) \f$;
     *      - \f$ \theta^{\indexUzawa+1} = \frac{1}{2} \theta^{\indexUzawa} \sqrt{4 + \left( \theta^{\indexUzawa} \right)^2} - \left( \theta^{\indexUzawa} \right)^2 \f$;
     *      - \f$ \beta^{\indexUzawa+1} = \theta^{\indexUzawa} \frac{1 - \theta^{\indexUzawa}}{\left( \theta^{\indexUzawa} \right)^2 + \theta^{\indexUzawa+1}} \f$;
     *      - \f$ \mathbf{y}^{\indexUzawa+1} = \mathbf{l}^{\indexUzawa+1} + \beta^{\indexUzawa+1} \left( \mathbf{l}^{\indexUzawa+1} - \mathbf{l}^{\indexUzawa} \right) \f$;
     *      - If (\f$ \mathbf{dg}^{\indexUzawa} \cdot \left( \mathbf{l}^{\indexUzawa+1} - \mathbf{l}^{\indexUzawa} \right) > 0 \f$ )
     *          - \f$ \mathbf{y}^{\indexUzawa+1} = \mathbf{l}^{\indexUzawa+1} \f$;
     *          - \f$ \theta^{\indexUzawa+1} = 1 \f$;
     *
     *      - \f$ \indexUzawa++ \f$.
     *
     * The projection depends on the problem.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t = DryWithoutFriction>
    class apgd_ar: public projection<problem_t>
    {
    protected:
        /**
         * @brief Constructor.
         *
         * @param max_iter [in] Maximal number of iterations.
         * @param rho [in] Step for the gradient descent.
         * @param tol_dg [in] Tolerance for \f$ \mathbf{dg} \f$ criterion.
         * @param tol_l [in] Tolerance for \f$ \mathbf{l} \f$ criterion.
         * @param verbose [in] Whether to compute and print the function cost.
         */
        apgd_ar(std::size_t max_iter, double rho, double tol_dg, double tol_l, bool verbose);
        /**
         * @brief Gradient descent algorithm.
         *
         * @param A [in] Matrix \f$ \mathbb{A} \f$.
         * @param descr [in] Structure specifying \f$ \mathbb{A} \f$ properties. 
         * @param c [in] Vector \f$ \mathbf{e} \f$.
         * @param l [out] vector \f$ \mathbf{l} \f$.
         *
         * @return Number of iterations the algorithm needed to converge.
         */
        std::size_t projection(const sparse_matrix_t& A, const struct matrix_descr& descr, const xt::xtensor<double, 1>& c, xt::xtensor<double, 1>& l);
    private:
        /**
         * @brief Maximal number of iterations.
         */
        std::size_t m_max_iter;
        /**
         * @brief Step for the gradient descent.
         */
        double m_rho;
        /**
         * @brief Tolerance for \f$ \mathbf{dg} \f$ criterion (unused).
         */
        double m_tol_dg;
        /**
         * @brief Tolerance for \f$ \mathbf{l} \f$ criterion.
         */
        double m_tol_l;
        /**
         * @brief Whether to compute and print the function cost.
         */
        bool m_verbose;

        /**
         * @brief Value indicating whether the operation was successful or not, and why.
         */
        sparse_status_t m_status;
        /**
         * @brief Vector \f$ \mathbf{dg}^{\indexUzawa} \f$.
         */
        xt::xtensor<double, 1> m_dg;
        /**
         * @brief Vector \f$ \mathbb{A} \mathbf{l}^{\indexUzawa+1} + \mathbf{e} \f$.
         */
        xt::xtensor<double, 1> m_uu;
        /**
         * @brief Vector \f$ \mathbf{y}^{\indexUzawa+1} \f$.
         */
        xt::xtensor<double, 1> m_y;
        /**
         * @brief Vector \f$ \mathbf{l}^{\indexUzawa} \f$.
         */
        xt::xtensor<double, 1> m_l_old;
        /**
         * @brief Vector \f$ \mathbf{l}^{\indexUzawa} \f$.
         */
        xt::xtensor<double, 1> m_lambda_prev;
    };

    template<class problem_t>
    apgd_ar<problem_t>::apgd_ar(std::size_t max_iter, double rho, double tol_dg, double tol_l, bool verbose)
    : projection<problem_t>()
    , m_max_iter(max_iter)
    , m_rho(rho)
    , m_tol_dg(tol_dg)
    , m_tol_l(tol_l)
    , m_verbose(verbose)
    {}

    template<class problem_t>
    std::size_t apgd_ar<problem_t>::projection(const sparse_matrix_t& A, const struct matrix_descr& descr, const xt::xtensor<double, 1>& c, xt::xtensor<double, 1>& l)
    {
        PLOG_INFO << "Projection: APGD-AR";
        std::size_t iter = 0;
        double theta_old = 1.;
        m_y = l;
        while (iter < m_max_iter)
        {
            xt::noalias(m_l_old) = l;
            // dg = A*y+c
            xt::noalias(m_dg) = c;
            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, m_y.data(), 1., m_dg.data());
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for dg = A*y+dg: " << m_status;

            xt::noalias(l) = this->projection_cone(m_y - m_rho * m_dg);
            double theta = 0.5*(theta_old*std::sqrt(4.+theta_old*theta_old) - theta_old*theta_old);
            double beta = theta_old*(1. - theta_old)/(theta_old*theta_old + theta);
            m_y = l + beta*(l - m_l_old);
            // double norm_dg = xt::amax(xt::abs(m_dg))(0);
            // double norm_l = xt::amax(xt::abs(l))(0);
            // double cmax = double((xt::amin(m_dg))(0));
            double diff_lambda = xt::amax(xt::abs(l - m_l_old))(0) / (xt::amax(xt::abs(m_l_old))(0) + 1.);

            if (m_verbose)
            {
                // uu = A*l + c
                xt::noalias(m_uu) = c;
                m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, l.data(), 1., m_uu.data());
                PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for uu = A*l+c: " << m_status;
                double constraint = double((xt::amin(m_uu))(0));
                // cout = 1./2.*l^T*A*l
                double cout;
                m_status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1./2., A, descr, l.data(), 0., m_uu.data(), &cout);
                PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_dotmv for cout = 1/2*l^T*A*l: " << m_status;
                PLOG_VERBOSE << constraint << "  " << cout + xt::linalg::dot(c, l)(0);
            }

            // if (norm_dg < m_tol_dg || norm_l < m_tol_l || cmax > -m_tol_dg)
            if (diff_lambda < m_tol_l)
            {
                return iter+1;
            }

            if (xt::linalg::dot(m_dg, l - m_l_old)(0) > 0.)
            {
                m_y = l;
                theta = 1.;
            }

            theta_old = theta;
            iter++;
        }
        PLOG_ERROR << "Uzawa does not converge";
        return iter;
    }
}
#endif
