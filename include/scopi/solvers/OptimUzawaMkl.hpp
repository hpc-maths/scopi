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

namespace scopi{
    template<std::size_t dim>
    class OptimUzawaMkl: public OptimUzawaBase<OptimUzawaMkl<dim>, dim>
                       , public MatrixOptimSolver<OptimUzawaMkl<dim>, dim>
    {
    public:
        OptimUzawaMkl(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
        void setup_impl(const std::vector<neighbor<dim>>& contacts);
        void tear_down_impl();

        void gemv_inv_P_impl();
        void gemv_A_impl(const std::vector<neighbor<dim>>& contacts);
        void gemv_transpose_A_impl(const std::vector<neighbor<dim>>& contacts);

    private:
        using base_type = OptimUzawaBase<OptimUzawaMkl<dim>, dim>;
        void print_csr_matrix(const sparse_matrix_t);

        sparse_matrix_t m_A;
        struct matrix_descr m_descrA;
        std::vector<MKL_INT> m_A_coo_row;
        std::vector<MKL_INT> m_A_coo_col;
        std::vector<double> m_A_coo_val;
        sparse_matrix_t m_inv_P;
        struct matrix_descr m_descr_inv_P;
        sparse_status_t m_status;

    };

    template<std::size_t dim>
    OptimUzawaMkl<dim>::OptimUzawaMkl(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr)
    : OptimUzawaBase<OptimUzawaMkl<dim>, dim>(particles, dt, Nactive, active_ptr)
    , MatrixOptimSolver<OptimUzawaMkl<dim>, dim>(particles, dt, Nactive, active_ptr)
    {}

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::setup_impl(const std::vector<scopi::neighbor<dim>>& contacts)
    {
        PLOG_NONE << "OptimUzawaMkl::setup_impl";
        // constraint matrix
        this->create_matrix_constraint_coo(contacts, m_A_coo_row, m_A_coo_col, m_A_coo_val, 0);

        sparse_matrix_t A_coo;
        m_status =  mkl_sparse_d_create_coo(&A_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           contacts.size(), // number of rows
                                           6*base_type::m_Nactive, // number of cols
                                           m_A_coo_val.size(), // number of non-zero elements
                                           m_A_coo_row.data(),
                                           m_A_coo_col.data(),
                                           m_A_coo_val.data());
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

        // mass matrix
        std::vector<MKL_INT> invP_csr_row;
        std::vector<MKL_INT> invP_csr_col;
        std::vector<double> invP_csr_val;
        invP_csr_col.reserve(6*base_type::m_Nactive);
        invP_csr_row.reserve(6*base_type::m_Nactive+1);
        invP_csr_val.reserve(6*base_type::m_Nactive);

        for (std::size_t i = 0; i < base_type::m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*i + d);
                invP_csr_col.push_back(3*i + d);
                invP_csr_val.push_back(1./this->m_mass); // TODO: add mass into particles
            }
        }
        for (std::size_t i = 0; i < base_type::m_Nactive; ++i)
        {
            for (std::size_t d  =0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*base_type::m_Nactive + 3*i + d);
                invP_csr_col.push_back(3*base_type::m_Nactive + 3*i + d);
                invP_csr_val.push_back(1./this->m_moment);
            }
        }
        invP_csr_row.push_back(6*base_type::m_Nactive);

        m_status = mkl_sparse_d_create_csr(&m_inv_P,
                                           SPARSE_INDEX_BASE_ZERO,
                                           6*base_type::m_Nactive, // number of rows
                                           6*base_type::m_Nactive, // number of cols
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

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::gemv_inv_P_impl()
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, this->m_U.data(), 0., this->m_U.data()); // U = - P^-1 * U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << m_status;
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::gemv_A_impl(const std::vector<neighbor<dim>>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_A, m_descrA, this->m_U.data(), 1., this->m_R.data()); // R = - A * U + R
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for R = - A * U + R: " << m_status;
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::gemv_transpose_A_impl(const std::vector<neighbor<dim>>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_A, m_descrA, this->m_L.data(), 1., this->m_U.data()); // U = A^T * L + U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << m_status;
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::tear_down_impl()
    {
        mkl_sparse_destroy ( m_inv_P );
        mkl_sparse_destroy ( m_A );
        m_A_coo_row.clear();
        m_A_coo_col.clear();
        m_A_coo_val.clear();
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::print_csr_matrix(const sparse_matrix_t A)
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
}
#endif
