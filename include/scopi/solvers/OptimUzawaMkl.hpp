#pragma once

#ifdef SCOPI_USE_MKL
#include "OptimUzawaBase.hpp"
#include "mkl_service.h"
#include "mkl_spblas.h"
#include "../problems/DryWithoutFriction.hpp"
#include <stdio.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    template<class problem_t>
    class OptimUzawaMkl;

    template<>
    class OptimParams<OptimUzawaMkl> : public OptimParamsUzawaBase
    {};

    template <class problem_t = DryWithoutFriction>
    class OptimUzawaMkl: public OptimUzawaBase<OptimUzawaMkl<problem_t>, problem_t>
    {
    public:
        using base_type = OptimUzawaBase<OptimUzawaMkl<problem_t>, problem_t>;

        template <std::size_t dim>
        OptimUzawaMkl(std::size_t nparts, double dt, const scopi_container<dim>& particles, OptimParams<OptimUzawaMkl>& optim_params);
        ~OptimUzawaMkl();

        template <std::size_t dim>
        void init_uzawa_impl(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts);
        void finalize_uzawa_impl();

        template <std::size_t dim>
        void gemv_inv_P_impl(const scopi_container<dim>& particles, problem_t& problem);

        template <std::size_t dim>
        void gemv_A_impl(const scopi_container<dim>& particles,
                         const std::vector<neighbor<dim>>& contacts,
                         problem_t& problem);

        template <std::size_t dim>
        void gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts,
                                   problem_t& problem);

    private:
        void print_csr_matrix(const sparse_matrix_t);

        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<2>& particles);
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<3>& particles);

        sparse_matrix_t m_A;
        struct matrix_descr m_descrA;
        std::vector<MKL_INT> m_A_coo_row;
        std::vector<MKL_INT> m_A_coo_col;
        std::vector<double> m_A_coo_val;
        sparse_matrix_t m_inv_P;
        struct matrix_descr m_descr_inv_P;
        sparse_status_t m_status;

    };

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::init_uzawa_impl(const scopi_container<dim>& particles,
                                        const std::vector<scopi::neighbor<dim>>& contacts)
    {
        this->create_matrix_constraint_coo(particles, contacts, 0);

        sparse_matrix_t A_coo;
        m_status =  mkl_sparse_d_create_coo(&A_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           this->number_row_matrix(contacts), // number of rows
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
    }

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::finalize_uzawa_impl()
    {
        m_status = mkl_sparse_destroy ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix A: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::gemv_inv_P_impl(const scopi_container<dim>&, problem_t&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, this->m_U.data(), 0., this->m_U.data()); // U = - P^-1 * U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::gemv_A_impl(const scopi_container<dim>&,
                                               const std::vector<neighbor<dim>>&,
                                               problem_t&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_A, m_descrA, this->m_U.data(), 1., this->m_R.data()); // R = - A * U + R
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for R = - A * U + R: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::gemv_transpose_A_impl(const scopi_container<dim>&,
                                                         const std::vector<neighbor<dim>>&,
                                                         problem_t&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_A, m_descrA, this->m_L.data(), 1., this->m_U.data()); // U = A^T * L + U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    OptimUzawaMkl<problem_t>::OptimUzawaMkl(std::size_t nparts, double dt, const scopi_container<dim>& particles, OptimParams<OptimUzawaMkl>& optim_params)
    : base_type(nparts, dt, optim_params)
    {
        std::vector<MKL_INT> invP_csr_row;
        std::vector<MKL_INT> invP_csr_col;
        std::vector<double> invP_csr_val;
        invP_csr_col.reserve(6*nparts);
        invP_csr_row.reserve(6*nparts+1);
        invP_csr_val.reserve(6*nparts);

        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*i + d);
                invP_csr_col.push_back(3*i + d);
                invP_csr_val.push_back(1./particles.m()(active_offset + i));
            }
        }
        set_moment_matrix(nparts, invP_csr_row, invP_csr_col, invP_csr_val, particles);
        invP_csr_row.push_back(6*nparts);

        m_status = mkl_sparse_d_create_csr(&m_inv_P,
                                           SPARSE_INDEX_BASE_ZERO,
                                           6*nparts, // number of rows
                                           6*nparts, // number of cols
                                           invP_csr_row.data(),
                                           invP_csr_row.data()+1,
                                           invP_csr_col.data(),
                                           invP_csr_val.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_csr for matrix invP: " << m_status;

        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_DIAGONAL;
        m_descr_inv_P.diag = SPARSE_DIAG_NON_UNIT;

        m_status = mkl_sparse_set_mv_hint(m_inv_P, SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix invP: " << m_status;

        m_status = mkl_sparse_optimize ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix invP: " << m_status;
    }

    template <class problem_t>
    OptimUzawaMkl<problem_t>::~OptimUzawaMkl()
    {
        m_status = mkl_sparse_destroy ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix P^-1: " << m_status;
    }

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::print_csr_matrix(const sparse_matrix_t A)
    {
        MKL_INT* csr_row_begin_ptr = NULL;
        MKL_INT* csr_row_end_ptr = NULL;
        MKL_INT* csr_col_ptr = NULL;
        double* csr_val_ptr = NULL;
        sparse_index_base_t indexing;
        MKL_INT nbRows;
        MKL_INT nbCols;
        m_status = mkl_sparse_d_export_csr(A,
                                           &indexing,
                                           &nbRows,
                                           &nbCols,
                                           &csr_row_begin_ptr,
                                           &csr_row_end_ptr,
                                           &csr_col_ptr,
                                           &csr_val_ptr);

        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_export_csr: " << m_status;

        std::cout << "\nMatrix with " << nbRows << " rows and " << nbCols << " columns\n";
        std::cout << "RESULTANT MATRIX:\nrow# : (column, value) (column, value)\n";
        int ii = 0;
        for( int i = 0; i < nbRows; i++ )
        {
            std::cout << "row#" << i << ": ";
            for(MKL_INT j = csr_row_begin_ptr[i]; j < csr_row_end_ptr[i]; j++ )
            {
                std::cout << " (" << csr_col_ptr[ii] << ", " << csr_val_ptr[ii] << ")";
                ii++;
            }
            std::cout << std::endl;
        }
        std::cout << "_____________________________________________________________________  \n" ;
    }

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::set_moment_matrix(std::size_t nparts,
                                          std::vector<MKL_INT>& invP_csr_row,
                                          std::vector<MKL_INT>& invP_csr_col,
                                          std::vector<double>& invP_csr_val,
                                          const scopi_container<2>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(0.);
            }
            invP_csr_row.push_back(3*nparts + 3*i + 2);
            invP_csr_col.push_back(3*nparts + 3*i + 2);
            invP_csr_val.push_back(1./particles.j()(active_offset + i));
        }
    }

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::set_moment_matrix(std::size_t nparts,
                                         std::vector<MKL_INT>& invP_csr_row,
                                         std::vector<MKL_INT>& invP_csr_col,
                                         std::vector<double>& invP_csr_val,
                                         const scopi_container<3>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(1./particles.j()(active_offset + i)(d));
            }
        }
    }
}
#endif
