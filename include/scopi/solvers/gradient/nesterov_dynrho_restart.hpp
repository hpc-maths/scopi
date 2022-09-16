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
    /**
     * @brief Accelerated Projected Gradient Descent algorithm with Adaptive Step and Adaptive Restart.
     *
     * See OptimProjectedGradient for the notations.
     * The algorithm is
     *  - \f$ k = 0 \f$;
     *  - \f$ \mathbf{l}^{k} = 0 \f$;
     *  - \f$ \mathbf{y}^{k} = 0 \f$;
     *  - \f$ \theta^{k} = 1 \f$.
     *  - \f$ \rho^{k} \f$ given;
     *  - \f$ L^{k} = \frac{1}{\rho^{k}} \f$;
     *  - While (\f$  \frac{||\mathbf{l}^{k} - \mathbf{l}^{k-1}||}{||\\mathbf{l}^{k}||+1} \le tol\_l \f$)
     *      - \f$ \mathbf{dg}^{k} = \mathbb{A} \mathbf{y}^{k} + \mathbf{e} \f$;
     *      - \f$ \mathbf{l}^{k+1} = \max \left (\mathbf{y}^{k} - \rho^{k} \mathbf{dg}^{k}, 0 \right) \f$;
     *      - While (\f$ \frac{1}{2} \mathbf{l}^{k+1} \cdot \mathbb{A} \mathbf{l}^{k+1} + \mathbf{e} \cdot \mathbf{l}^{k+1} > \frac{1}{2} \mathbf{y}^{k+1} \cdot \mathbb{A} \mathbf{y}{k+1} + \mathbf{e} \cdot \mathbf{y}^{k+1} + \mathbf{dg}^{k} \cdot \left( \mathbf{l}^{k+1} - \mathbf{y}^{k+1} \right) + \frac{1}{2} L^{k} \left( \mathbf{l}^{k+1} - \mathbf{y}^{k+1} \right) \cdot \left( \mathbf{l}^{k+1} - \mathbf{y}^{k+1} \right) \f$)
     *          - \f$ L^{k} = 2 L^{k} \f$;
     *          - \f$ \rho^{k} = \frac{1}{L^{k}} \f$;
     *          - \f$ \mathbf{l}^{k+1} = \max \left( \mathbf{y}^{k} - \rho^{k} \mathbf{dg}^{k}, 0 \right) \f$;
     *
     *      - \f$ \theta^{k+1} = \frac{1}{2} \theta^{k} \sqrt{4 + \left( \theta^{k} \right)^2} - \left( \theta^{k} \right)^2 \f$;
     *      - \f$ \beta^{k+1} = \theta^{k} \frac{1 - \theta^{k}}{\left( \theta^{k} \right)^2 + \theta^{k+1}} \f$;
     *      - \f$ \mathbf{y}^{k+1} = \mathbf{l}^{k+1} + \beta^{k+1} \left( \mathbf{l}^{k+1} - \mathbf{l}^{k} \right) \f$;
     *      - If (\f$ \mathbf{dg}^{k} \cdot \left( \mathbf{l}^{k+1} - \mathbf{l}^{k} \right) > 0 \f$ )
     *          - \f$ \mathbf{y}^{k+1} = \mathbf{l}^{k+1} \f$;
     *          - \f$ \theta^{k+1} = 1 \f$;
     *
     *      - \f$ k++ \f$.
     *
     * @tparam projection_t Projection on admissible velocities.
     */
    template<class projection_t = projection_max>
    class nesterov_dynrho_restart: public projection_t
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
        nesterov_dynrho_restart(std::size_t max_iter, double rho, double tol_dg, double tol_l, bool verbose);
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
         * @brief Initial guess for the step for the gradient descent.
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
         * @brief Step for the gradient descent.
         */
        double m_rho_init;

        /**
         * @brief Value indicating whether the operation was successful or not, and why.
         */
        sparse_status_t m_status;
        /**
         * @brief Vector \f$ \mathbf{dg}^{k} \f$.
         */
        xt::xtensor<double, 1> m_dg;
        /**
         * @brief Vector \f$ \mathbb{A} \mathbf{l}^{k+1} + \mathbf{e} \f$.
         */
        xt::xtensor<double, 1> m_uu;
        /**
         * @brief Vector \f$ \mathbf{y}^{k+1} \f$.
         */
        xt::xtensor<double, 1> m_y;
        /**
         * @brief Vector \f$ \mathbf{l}^{k} \f$.
         */
        xt::xtensor<double, 1> m_l_old;
        /**
         * @brief Temporary vector used to compute \f$ \mathbf{l}^T \cdot \mathbb{A} \mathbf{l} \f$ and \f$ \mathbf{y}^T \cdot \mathbb{A} \mathbf{y} \f$.
         */
        xt::xtensor<double, 1> m_tmp;
        /**
         * @brief Vector \f$ \mathbf{l}^{k} \f$.
         */
        xt::xtensor<double, 1> m_lambda_prev;
    };

    template<class projection_t>
    nesterov_dynrho_restart<projection_t>::nesterov_dynrho_restart(std::size_t max_iter, double rho, double tol_dg, double tol_l, bool verbose)
    : projection_t()
    , m_max_iter(max_iter)
    , m_rho(rho)
    , m_tol_dg(tol_dg)
    , m_tol_l(tol_l)
    , m_verbose(verbose)
    , m_rho_init(rho)
    {}

    template<class projection_t>
    std::size_t nesterov_dynrho_restart<projection_t>::projection(const sparse_matrix_t& A, const struct matrix_descr& descr, const xt::xtensor<double, 1>& c, xt::xtensor<double, 1>& l)
    {
        PLOG_INFO << "Projection: Nesterov with adaptive step size and restart";
        std::size_t iter = 0;
        double theta_old = 1.;
        m_y = l;
        m_l_old = l;
        m_rho = m_rho_init;
        double lipsch = 1./m_rho;
        m_tmp.resize({l.size()});
        while (iter < m_max_iter)
        {
            xt::noalias(m_lambda_prev) = l;
            // dg = A*y+c
            xt::noalias(m_dg) = c;
            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, m_y.data(), 1., m_dg.data());
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for dg = A*y+dg: " << m_status;

            xt::noalias(l) = this->projection_cone(m_y - m_rho * m_dg);

            std::size_t cc = 0;
            // lAl = l^T*A*l
            double lAl;
            m_status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, l.data(), 0., m_tmp.data(), &lAl);
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_dotmv for lAl = l^T*A*l: " << m_status;
            double cl = xt::linalg::dot(c, l)(0);
            // yAy = y^T*A*y
            double yAy;
            m_status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, m_y.data(), 0., m_tmp.data(), &yAy);
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_dotmv for yAy = y^T*A*y: " << m_status;
            double cy = xt::linalg::dot(c, m_y)(0);
            double dgly = xt::linalg::dot(m_dg, l - m_y)(0);
            double lyly = xt::linalg::dot(l - m_y, l - m_y)(0);

            while ((0.5*lAl + cl > 0.5*yAy + cy + dgly + lipsch/2.*lyly) && (cc < 10))
            {
                lipsch *= 2.;
                m_rho = 1./lipsch;
                xt::noalias(l) = this->projection_cone(m_y - m_rho * m_dg);

                cc++;
                // lAl = l^T*A*l
                m_status = mkl_sparse_d_dotmv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, l.data(), 0., m_tmp.data(), &lAl);
                PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_dotmv for lAl = l^TA*l: " << m_status;
                cl = xt::linalg::dot(c, l)(0);
                dgly = xt::linalg::dot(m_dg, l - m_y)(0);
                lyly = xt::linalg::dot(l - m_y, l - m_y)(0);
            }

            double theta = 0.5*(theta_old*std::sqrt(4.+theta_old*theta_old) - theta_old*theta_old);
            double beta = theta_old*(1. - theta_old)/(theta_old*theta_old + theta);
            m_y = l + beta*(l - m_l_old);
            // double norm_dg = xt::amax(xt::abs(m_dg))(0);
            // double norm_l = xt::amax(xt::abs(l))(0);
            // double cmax = double((xt::amin(m_dg))(0));
            double diff_lambda = xt::amax(xt::abs(l - m_lambda_prev))(0) / (xt::amax(xt::abs(m_lambda_prev))(0) + 1.);

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

            if (diff_lambda < m_tol_l)
            // if (norm_dg < m_tol_dg || norm_l < m_tol_l || cmax > -m_tol_dg)
            {
                return iter+1;
            }

            if (xt::linalg::dot(m_dg, l - m_l_old)(0) > 0.)
            {
                m_y = l;
                theta = 1.;
            }

            m_l_old = l;
            theta_old = theta;
            lipsch *= 0.97; // 0.9 in the paper
            m_rho = 1./lipsch;
            iter++;
        }
        PLOG_ERROR << "Uzawa does not converge";
        return iter;
    }
}
#endif
