#pragma once

#ifdef SCOPI_USE_MKL
#include "OptimUzawaBase.hpp"
#include "MatrixOptimSolver.hpp"
#include "mkl_service.h"
#include "mkl_spblas.h"
#include <stdio.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    class OptimUzawaMkl: public OptimUzawaBase<OptimUzawaMkl>
                       , public MatrixOptimSolver
    {
    public:
        using base_type = OptimUzawaBase<OptimUzawaMkl>;

        OptimUzawaMkl(std::size_t nparts, double dt, double mu = 0.);
        ~OptimUzawaMkl();

        template <std::size_t dim>
        void init_uzawa_impl(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void gemv_inv_P_impl(const scopi_container<dim>& particles);

        template <std::size_t dim>
        void gemv_A_impl(const scopi_container<dim>& particles,
                         const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts);

    private:
        void print_csr_matrix(const sparse_matrix_t);

        sparse_matrix_t m_A;
        struct matrix_descr m_descrA;
        std::vector<MKL_INT> m_A_coo_row;
        std::vector<MKL_INT> m_A_coo_col;
        std::vector<double> m_A_coo_val;
        sparse_matrix_t m_inv_P;
        struct matrix_descr m_descr_inv_P;
        sparse_status_t m_status;
        bool should_destroy;

    };

    template<std::size_t dim>
    void OptimUzawaMkl::init_uzawa_impl(const scopi_container<dim>& particles,
                                        const std::vector<scopi::neighbor<dim>>& contacts)
    {
        if(should_destroy)
        {
            m_status = mkl_sparse_destroy ( m_A );
            PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix A: " << m_status;
        }

        this->create_matrix_constraint_coo(particles, contacts, 0);

        sparse_matrix_t A_coo;
        m_status =  mkl_sparse_d_create_coo(&A_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           contacts.size(), // number of rows
                                           6*this->m_nparts, // number of cols
                                           this->m_A_values.size(), // number of non-zero elements
                                           this->m_A_rows.data(),
                                           this->m_A_cols.data(),
                                           this->m_A_values.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_coo for matrix A: " << m_status;

        m_status = mkl_sparse_convert_csr(A_coo,
                                          SPARSE_OPERATION_NON_TRANSPOSE,
                                          &m_A);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_convert_csr for matrix A: " << m_status;

        m_descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_NON_TRANSPOSE, m_descrA, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_NON_TRANSPOSE: " << m_status;

        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_TRANSPOSE, m_descrA, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_TRANSPOSE: " << m_status;

        m_status = mkl_sparse_optimize ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix A: " << m_status;

        should_destroy = true;
    }

    template<std::size_t dim>
    void OptimUzawaMkl::gemv_inv_P_impl(const scopi_container<dim>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, this->m_U.data(), 0., this->m_U.data()); // U = - P^-1 * U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << m_status;
    }

    template<std::size_t dim>
    void OptimUzawaMkl::gemv_A_impl(const scopi_container<dim>&,
                                         const std::vector<neighbor<dim>>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_A, m_descrA, this->m_U.data(), 1., this->m_R.data()); // R = - A * U + R
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for R = - A * U + R: " << m_status;
    }

    template<std::size_t dim>
    void OptimUzawaMkl::gemv_transpose_A_impl(const scopi_container<dim>&,
                                                   const std::vector<neighbor<dim>>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_A, m_descrA, this->m_L.data(), 1., this->m_U.data()); // U = A^T * L + U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << m_status;
    }
}
#endif
